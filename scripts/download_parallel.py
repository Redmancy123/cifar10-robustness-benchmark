"""Four bounded HTTP Range readers; full published MD5 is mandatory before extraction."""
import concurrent.futures
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import prepare_data as prep

CHUNK=8*1024*1024


def parallel_download(name,url,expected):
    destination=prep.RAW/name
    if destination.exists():
        if prep.md5(destination)!=expected:raise RuntimeError('Archive MD5 mismatch')
        return destination
    headers={'User-Agent':'CIFAR-coursework/1.0'}
    # Sizes checked against the official response headers; each GET verifies its full range/total.
    size={'cifar-10-python.tar.gz':170498071,'CIFAR-10-C.tar':2918471680}[name]
    if name=='cifar-10-python.tar.gz':
        url='https://cave.cs.toronto.edu/kriz/cifar-10-python.tar.gz'
    parts=prep.RAW/(name+'.parts');parts.mkdir(exist_ok=True)
    old=destination.with_suffix(destination.suffix+'.partial')
    # Reuse only complete sequential chunks. Final merged MD5 still verifies every reused byte.
    if old.exists():
        with old.open('rb') as f:
            for index in range(old.stat().st_size//CHUNK):
                chunk=f.read(CHUNK)
                path=parts/('%05d.part'%index)
                if not path.exists() and len(chunk)==CHUNK:path.write_bytes(chunk)
    count=(size+CHUNK-1)//CHUNK
    def fetch(index):
        start=index*CHUNK;end=min(size,start+CHUNK)-1;path=parts/('%05d.part'%index)
        if path.exists() and path.stat().st_size==end-start+1:return index
        for attempt in range(4):
            temp=path.with_suffix('.incomplete')
            try:
                request=urllib.request.Request(url,headers=dict(headers,Range='bytes=%d-%d'%(start,end)))
                with urllib.request.urlopen(request,timeout=60) as r:
                    if r.status!=206 or r.headers.get('Content-Range')!='bytes %d-%d/%d'%(start,end,size):
                        raise RuntimeError('Server did not honor exact Range')
                    with temp.open('wb') as f:
                        remaining=end-start+1
                        while remaining:
                            data=r.read(min(1024*1024,remaining))
                            if not data:raise RuntimeError('Truncated HTTP Range')
                            f.write(data);remaining-=len(data)
                        if r.read(1):raise RuntimeError('Oversized HTTP Range')
                if temp.stat().st_size!=end-start+1:raise RuntimeError('Part length mismatch')
                os.replace(temp,path);return index
            except Exception:
                if attempt==3:raise
                time.sleep(2**attempt)
    fetch(0)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures=[executor.submit(fetch,index) for index in range(1,count)]
        try:
            for done,future in enumerate(concurrent.futures.as_completed(futures),2):
                future.result()
                print(name,'verified-length parts',done,'/',count,flush=True)
        except Exception:
            for future in futures:future.cancel()
            raise
    assembled=destination.with_suffix(destination.suffix+'.assembling')
    with assembled.open('wb') as out:
        for index in range(count):
            with (parts/('%05d.part'%index)).open('rb') as source:shutil.copyfileobj(source,out,4*1024*1024)
    if assembled.stat().st_size!=size or prep.md5(assembled)!=expected:
        raise RuntimeError('Merged archive MD5 failed; no extraction will occur')
    os.replace(assembled,destination)
    return destination


if __name__=='__main__':
    prep.download=parallel_download
    prep.main()
