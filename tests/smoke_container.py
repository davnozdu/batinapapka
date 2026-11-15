"""Exercise the actual built image, HTTP API and Cyberdrop v10 invocation in CI."""
import base64
import json
import subprocess
import time
import urllib.error
import urllib.request


def docker(*args):
    return subprocess.check_output(['docker', *args], text=True).strip()


container = docker('run', '-d', '-e', 'ADMIN_USER=smoke', '-e', 'ADMIN_PASSWORD=smoke-test-password',
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
    assert 'Unknown option' not in log and 'No such option' not in log and 'Traceback' not in log, log
    assert 'Cyberdrop' in log or 'cyberdrop' in log, log
    docker('restart',container)
    for _ in range(60):
        try:
            assert json.loads(request('/api/jobs'))[0]['id']==ident
            break
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    else:
        raise AssertionError('Queue did not survive container restart')
    print('Container HTTP, actual Cyberdrop CLI and persistence smoke checks passed')
finally:
    print(docker('logs',container))
    docker('rm','-f',container)
