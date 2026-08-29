import builtins,hashlib,os,stat,sys
s=sys.argv[1]; p=os.path.abspath(sys.argv[2]); e=sys.argv[3]
if not (len(s)==64 and all(c in "0123456789abcdef" for c in s) and os.path.isabs(sys.argv[2]) and p==sys.argv[2] and len(e)==64 and all(c in "0123456789abcdef" for c in e)): raise SystemExit("operator_preloader_shim_arguments_rejected")
f=os.open(p,os.O_RDONLY|getattr(os,"O_BINARY",0)|getattr(os,"O_NOFOLLOW",0))
try:
 b=os.fstat(f); q=os.lstat(p)
 if not (stat.S_ISREG(b.st_mode) and b.st_nlink==q.st_nlink==1 and (b.st_dev,b.st_ino)==(q.st_dev,q.st_ino) and 0<b.st_size<=4194304): raise SystemExit("operator_preloader_shim_identity_rejected")
 r=b""; n=b.st_size
 while n:
  x=os.read(f,min(65536,n))
  if not x: raise SystemExit("operator_preloader_shim_short_read")
  r+=x; n-=len(x)
 if os.read(f,1): raise SystemExit("operator_preloader_shim_grew")
 a=os.fstat(f)
 if (b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)!=(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns): raise SystemExit("operator_preloader_shim_changed")
finally: os.close(f)
if hashlib.sha256(r).hexdigest()!=e: raise SystemExit("operator_preloader_shim_digest_rejected")
builtins._EXPLAINIVERSE_OPERATOR_SHIM_RECEIPT={"schema_version":1,"kind":"explainiverse-operator-preloader-shim","shim_sha256":s,"preloader_path":p,"preloader_bytes":len(r),"preloader_sha256":e,"stable_descriptor_read":True,"compiled_verified_bytes_without_reopen":True}
sys.argv=[p,*sys.argv[4:]]
g={"__name__":"__main__","__file__":p,"__builtins__":builtins.__dict__}
exec(compile(r,p,"exec",dont_inherit=True),g,g)
