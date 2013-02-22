
function bout = unstdize(b,stobj)

bout = b./stobj.st';
if ~isfinite(bout(1))
    bout(1) = b(1)-stobj.mu(2:end)./stobj.st(2:end)*b(2:end);
end