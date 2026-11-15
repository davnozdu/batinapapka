"""Run the private cron script with bounded stdout/stderr retention."""
import codecs
import fcntl
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import subprocess
import sys
from log_retention import trim_append_file


def main():
    state = Path(os.environ.get('STATE_DIR', '/state'))
    with (state / '.cron-output.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return 75
        for name in ('cron.log','cron.log.1','cron.log.2'):
            trim_append_file(state / name, 5*1024*1024)
        handler = RotatingFileHandler(state / 'cron.log', maxBytes=5*1024*1024, backupCount=2)
        output = logging.getLogger('cron-output')
        output.setLevel(logging.INFO)
        output.addHandler(handler)
        try:
            with subprocess.Popen([str(state / '.run-renamer.sh')], stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT) as proc:
                decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
                for chunk in iter(lambda:proc.stdout.read1(8192), b''):
                    output.info('%s', decoder.decode(chunk))
                final = decoder.decode(b'', final=True)
                if final:
                    output.info('%s', final)
                return proc.wait()
        finally:
            output.removeHandler(handler)
            handler.close()


if __name__ == '__main__':
    sys.exit(main())
