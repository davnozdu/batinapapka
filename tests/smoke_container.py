"""Exercise the actual built image, HTTP API and Cyberdrop v10 invocation in CI."""
import base64
import json
import os
import subprocess
import time
import urllib.error
import urllib.request


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


container = docker('run', '-d', '-e', 'ADMIN_USER=smoke', '-e', 'ADMIN_PASSWORD=smoke-test-password',
                   '-e', 'SHELL_USER=smoke', '-e', 'SHELL_PASSWORD=smoke-test-password',
                   '-e', 'CRON_SCHEDULE=', '-e', 'CYBERDROP_AUTO_UPDATE=0',
                   '-p', '127.0.0.1::8080', 'batinapapka-cyberdrop:smoke')
try:
    port = docker('port', container, '8080/tcp').split(':')[-1]
    base = 'http://127.0.0.1:' + port
    auth = 'Basic ' + base64.b64encode(b'smoke:smoke-test-password').decode()
    def request(path, data=None):
        req = urllib.request.Request(base+path, data=data,
              headers={'Authorization':auth,'X-Admin-Request':'1',
                       'Content-Type':'application/x-www-form-urlencoded'})
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read()
    for _ in range(60):
        try:
            request('/api/jobs')
            break
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    else:
        raise AssertionError('Web service did not start')
    assert b'Cyberdrop' in request('/')
    bundled = docker('exec', container, '/opt/cyberdrop/venv/bin/cyberdrop-dl', '--version')
    print('Bundled Cyberdrop:', bundled)
    if os.environ.get('EXPECTED_CYBERDROP_VERSION'):
        assert bundled == os.environ['EXPECTED_CYBERDROP_VERSION'], bundled
    assert json.loads(request('/api/telegram')) == {'enabled':False,'configured':False,'allowed_user_ids':[]}
    # An older runtime venv from the previous image must not shadow the new bundle.
    docker('exec', '-u', 'smoke', container, 'python', '-c',
           "from pathlib import Path; import json,time; r=Path('/state/cyberdrop/runtime'); r.mkdir(exist_ok=True); "
           "p=r/'venv-10.15.3-1234abcd/bin/cyberdrop-dl'; p.parent.mkdir(parents=True); "
           "p.write_text('#!/bin/sh\\necho old-runtime-should-not-run\\n'); p.chmod(0o755); "
           "(r/'current').symlink_to(p.parent.parent.name,target_is_directory=True); "
           "(r/'status.json').write_text(json.dumps({'version':'10.15.3','checked':time.time()}))")
    identity = docker('exec', container, 'cat', '/state/ssh/ssh_host_ed25519_key.pub')
    # Reproduce an existing V9 installation: V10 used to fail despite an explicit config.
    docker('exec', '-u', 'smoke', container, 'python', '-c',
           "from pathlib import Path; p=Path('/state/cyberdrop/AppData/Configs/Default/settings.yaml'); "
           "p.parent.mkdir(parents=True,exist_ok=True); p.write_text('legacy_v9: true\\n')")
    ident=json.loads(request('/api/jobs',b'urls=https%3A%2F%2Fexample.invalid%2Fsmoke'))['id']
    for _ in range(120):
        job=json.loads(request('/api/jobs'))[0]
        if job['status'] in ('done','failed'):
            break
        time.sleep(1)
    else:
        raise AssertionError('Actual Cyberdrop process did not finish')
    log=json.loads(request(f'/api/jobs/{ident}/log'))['log']
    print(log[-12000:])
    assert 'old-runtime-should-not-run' not in log, log
    assert 'Migration Error' not in log and job['status'] == 'done', log
    assert 'Unknown option' not in log and 'No such option' not in log and 'Traceback' not in log, log
    assert 'Cyberdrop' in log or 'cyberdrop' in log, log
    assert docker('exec',container,'cat','/state/cyberdrop/AppData/Configs/Default/settings.yaml').startswith('legacy_v9')
    assert docker('exec',container,'readlink',f'/state/cyberdrop/jobs/{ident}/managed.log') == '/dev/null'
    docker('restart',container)
    port = docker('port', container, '8080/tcp').split(':')[-1]
    base = 'http://127.0.0.1:' + port
    for _ in range(60):
        try:
            assert json.loads(request('/api/jobs'))[0]['id']==ident
            break
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    else:
        raise AssertionError('Queue did not survive container restart')
    assert docker('exec', container, 'cat', '/state/ssh/ssh_host_ed25519_key.pub') == identity
    assert json.loads(request(f'/api/jobs/{ident}/delete',b''))['ok']
    assert json.loads(request('/api/jobs')) == []
    assert docker('exec',container,'python','-c',f"from pathlib import Path; print(any(Path('/state/cyberdrop/jobs').glob('{ident}*')))") == 'False'
    print('Container HTTP, SSH setup, actual Cyberdrop CLI, persistence and deletion checks passed')
finally:
    print(docker('logs',container))
    docker('rm','-f',container)
