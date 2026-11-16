"""Resolve a stable PyPI release and compare it with the published GHCR image."""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request

PYPI = 'https://pypi.org/pypi/cyberdrop-dl-patched/json'
VERSION = re.compile(r'\d+\.\d+\.\d+')
LABEL = 'io.batinapapka.cyberdrop.version'


def version_key(value):
    if not isinstance(value, str) or not VERSION.fullmatch(value):
        raise ValueError('Expected a stable Cyberdrop version in X.Y.Z format')
    return tuple(int(part) for part in value.split('.'))


def get_json(url, headers=None):
    request = urllib.request.Request(url, headers={'User-Agent':'BatinaPapka-upstream-check',
                                                   'Cache-Control':'no-cache', **(headers or {})})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def latest_stable(data):
    versions = [version for version, files in data['releases'].items()
                if VERSION.fullmatch(version) and any(not item.get('yanked', False) for item in files)]
    if not versions:
        raise ValueError('PyPI has no non-yanked stable Cyberdrop release')
    return max(versions, key=version_key)


def published_version(image):
    if not re.fullmatch(r'ghcr\.io/[a-z0-9_-]+/[a-z0-9_.-]+', image):
        raise ValueError('Expected ghcr.io/owner/image')
    name = image.removeprefix('ghcr.io/')
    token = get_json('https://ghcr.io/token?service=ghcr.io&scope=repository:'+name+':pull')['token']
    headers = {'Authorization':'Bearer '+token, 'Accept':','.join((
        'application/vnd.oci.image.index.v1+json', 'application/vnd.oci.image.manifest.v1+json',
        'application/vnd.docker.distribution.manifest.list.v2+json',
        'application/vnd.docker.distribution.manifest.v2+json'))}
    base = 'https://ghcr.io/v2/'+name
    try:
        manifest = get_json(base+'/manifests/latest', headers)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return ''
        raise
    if 'manifests' in manifest:
        descriptor = next(item for item in manifest['manifests']
                          if item.get('platform', {}).get('os') == 'linux' and
                          item['platform'].get('architecture') == 'amd64')
        manifest = get_json(base+'/manifests/'+descriptor['digest'], headers)
    config = get_json(base+'/blobs/'+manifest['config']['digest'], headers)
    version = (config.get('config', {}).get('Labels') or {}).get(LABEL, '')
    return version if VERSION.fullmatch(version) else ''


def needs_update(latest, published, force=False):
    return force or not published or version_key(latest) > version_key(published)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default='')
    parser.add_argument('--image', default='ghcr.io/davnozdu/batinapapka-cyberdrop')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--resolve-only', action='store_true')
    args = parser.parse_args()
    version = args.version or latest_stable(get_json(PYPI))
    version_key(version)
    published = '' if args.resolve_only else published_version(args.image)
    update = False if args.resolve_only else needs_update(version, published, args.force)
    result = {'version':version, 'published':published, 'update_needed':str(update).lower()}
    print(json.dumps(result))
    if os.environ.get('GITHUB_OUTPUT'):
        with Path(os.environ['GITHUB_OUTPUT']).open('a') as stream:
            for name, value in result.items():
                stream.write(f'{name}={value}\n')
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a') as stream:
            stream.write(f'Cyberdrop stable: **{version}**; published image: **{published or "unlabelled/missing"}**.\n\n')
            if not args.resolve_only:
                stream.write('Rebuild required.\n' if update else 'Already current; no image rebuild.\n')


if __name__ == '__main__':
    main()
