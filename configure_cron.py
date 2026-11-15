"""Write a private runner script rather than interpolating secrets into cron."""
import os
from pathlib import Path
import shlex
import sys


def main():
    state = Path(os.environ.get('STATE_DIR', '/state'))
    state.mkdir(parents=True, exist_ok=True)
    schedule = os.environ.get('CRON_SCHEDULE', '')
    user = os.environ.get('SHELL_USER') or 'root'
    target = Path('/etc/cron.d/batinapapka')
    target.unlink(missing_ok=True)
    if not schedule or not os.environ.get('BRAVE_API_KEY'):
        return
    if len(schedule.split()) != 5 or any(c in schedule for c in '\r\n%'):
        raise ValueError('CRON_SCHEDULE must contain five cron fields, without newlines or %')
    runner = state / '.run-renamer.sh'
    args = sys.argv[1:] or ['--recursive', '/videos']
    runner.write_text('#!/bin/sh\n' + 'cd ' + shlex.quote(str(state)) + '\n' +
                      'export BRAVE_API_KEY=' + shlex.quote(os.environ['BRAVE_API_KEY']) + '\n' +
                      'exec /usr/local/bin/python /app/batinapapka.py ' + shlex.join(args) + '\n')
    runner.chmod(0o700)
    import pwd
    account = pwd.getpwnam(user)
    os.chown(runner, account.pw_uid, account.pw_gid)
    # Cron invokes a fixed command. The script holds arguments/environment safely.
    Path('/usr/local/bin/run-scheduled-renamer').write_text(
        '#!/bin/sh\nexec ' + shlex.quote(str(runner)) + ' >> ' + shlex.quote(str(state / 'cron.log')) + ' 2>&1\n')
    Path('/usr/local/bin/run-scheduled-renamer').chmod(0o755)
    target.write_text(f'{schedule} {user} /usr/local/bin/run-scheduled-renamer\n')
    target.chmod(0o644)


if __name__ == '__main__':
    main()
