// Retain a camera stream across Streamlit component data updates.
const cameras = new WeakMap();
export default function(component) {
  const {parentElement, data} = component;
  const root = parentElement.querySelector('#camera');
  let state = cameras.get(root);
  if (!state) {
    const video = root.querySelector('video'), overlay = root.querySelector('#overlay');
    state = {video, overlay, stream:null, active:false, busy:false, seq:0, objects:[], frameTime:0,
             capture:document.createElement('canvas'), status:root.querySelector('#status')};
    cameras.set(root, state);
    state.updateQuality = () => {
      if (!state.active || !video.videoWidth || !video.videoHeight) return;
      // The negotiated video dimensions are authoritative, including portrait cameras.
      const width=video.videoWidth, height=video.videoHeight;
      root.querySelector('#stage').style.setProperty('--camera-aspect', `${width} / ${height}`);
      const maxSide=state.data.analysisSize||1280, scale=Math.min(1,maxSide/Math.max(width,height));
      const settings=state.stream?.getVideoTracks()[0]?.getSettings()||{};
      const fps=settings.frameRate ? ` · ${Math.round(settings.frameRate)} fps` : '';
      root.querySelector('#quality').textContent=`Camera: ${width} × ${height}${fps} · Phân tích: ${Math.round(width*scale)} × ${Math.round(height*scale)} · tối đa ${state.data.fps} khung/giây`;
    };
    video.onresize=state.updateQuality;
    state.stop = (announce=true) => {
      const session=state.session; state.session=null;
      state.active = false;
      state.stream?.getTracks().forEach(track => track.stop());
      state.stream = null; video.srcObject = null; state.busy=false; state.objects=[];
      clearTimeout(state.timer); cancelAnimationFrame(state.animation);
      overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
      root.querySelector('#start').disabled = false; root.querySelector('#stop').disabled = true;
      root.querySelector('#placeholder').hidden = false;
      root.querySelector('#quality').textContent='Sử dụng khung hình theo camera của thiết bị.';
      state.drawSignature=null;
      state.status.textContent = 'Camera đã dừng.';
      if (announce) state.component.setStateValue('event', {session, active:false});
    };
    state.draw = () => {
      if (!state.active) return;
      const width = overlay.clientWidth, height = overlay.clientHeight;
      const dpr=window.devicePixelRatio||1, fresh=performance.now()-state.frameTime<2500;
      const signature=[width,height,dpr,state.frameTime,state.data.selected,fresh].join(':');
      if(signature===state.drawSignature){state.animation=requestAnimationFrame(state.draw);return;}
      state.drawSignature=signature;
      const backingWidth=Math.round(width*dpr),backingHeight=Math.round(height*dpr);
      if(overlay.width!==backingWidth || overlay.height!==backingHeight){overlay.width=backingWidth;overlay.height=backingHeight;}
      const ctx = overlay.getContext('2d');
      ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,width,height);
      const scale = Math.min(width/(state.frameWidth||1),height/(state.frameHeight||1));
      const dx=(width-(state.frameWidth||0)*scale)/2, dy=(height-(state.frameHeight||0)*scale)/2;
      if (fresh) for (const object of state.objects) {
        const [x1,y1,x2,y2]=object.box;
        ctx.strokeStyle = object.id===state.data.selected ? '#ffe077' : '#40e6bd'; ctx.lineWidth=3;
        ctx.strokeRect(dx+x1*scale,dy+y1*scale,(x2-x1)*scale,(y2-y1)*scale);
        ctx.font='bold 15px Arial'; const text=`#${object.id} ${object.label} ${Math.round(object.confidence*100)}%`;
        const x=dx+x1*scale, y=Math.max(22,dy+y1*scale);
        ctx.fillStyle='#092b30'; ctx.fillRect(x,y-22,ctx.measureText(text).width+14,22);
        ctx.fillStyle='white'; ctx.fillText(text,x+7,y-6);
      }
      state.animation=requestAnimationFrame(state.draw);
    };
    state.tick = () => {
      if (!state.active) return;
      if (state.busy && performance.now()-state.sentAt>30000) {
        state.stop(); state.status.textContent='Mất phản hồi từ app. Bật lại camera để kết nối.'; return;
      }
      if (!state.busy && video.readyState>=2) {
        const capture=state.capture;
        const scale=Math.min(1,(state.data.analysisSize||1280)/Math.max(video.videoWidth,video.videoHeight));
        const width=Math.max(1,Math.round(video.videoWidth*scale)),height=Math.max(1,Math.round(video.videoHeight*scale));
        if(capture.width!==width || capture.height!==height){capture.width=width;capture.height=height;}
        capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);
        let jpeg=capture.toDataURL('image/jpeg',.9).split(',')[1];
        // Stay within the server's encoded payload bound even for a noisy, detailed frame.
        if(jpeg.length>2000000)jpeg=capture.toDataURL('image/jpeg',.7).split(',')[1];
        state.busy=true;state.sentAt=performance.now();state.seq++;
        state.component.setStateValue('frame', {session:state.session,seq:state.seq,
          jpeg});
      }
      state.timer=setTimeout(state.tick,1000/state.data.fps);
    };
    const devices=root.querySelector('#devices');
    const start=async () => {
      state.stop(false); root.querySelector('#start').disabled=true;
      root.querySelector('#stop').disabled=false;
      state.status.textContent='Đang xin quyền truy cập camera…';
      // A token prevents a pending permission dialog from reopening a stopped/unmounted stream.
      const token=state.session=crypto.randomUUID();
      try {
        if(!navigator.mediaDevices?.getUserMedia) throw new Error('secure');
        // Ask for a native high-resolution stream; the browser negotiates available modes.
        const constraints={width:{ideal:3840},height:{ideal:2160},resizeMode:{ideal:'none'}};
        if(devices.value)constraints.deviceId={exact:devices.value};else constraints.facingMode={ideal:'environment'};
        const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:constraints});
        if(token!==state.session || !root.isConnected){stream.getTracks().forEach(t=>t.stop());return;}
        state.stream=stream;
        const track=stream.getVideoTracks()[0];
        // Capabilities are ranges, not a promise that the largest width/height pair is a mode.
        // Use ideals and retain the working stream if renegotiation is unsupported.
        try {
          const capabilities=track?.getCapabilities?.()||{};
          if(track?.applyConstraints && capabilities.width?.max && capabilities.height?.max){
            const preferred={...constraints,width:{ideal:capabilities.width.max},height:{ideal:capabilities.height.max}};
            await track.applyConstraints(preferred);
          }
          // Negotiate FPS only after resolution, otherwise a fast low-resolution mode can win.
          // Keep the negotiated dimensions fixed while asking for the best frame rate at that size.
          const native=track?.getSettings?.()||{};
          if(track?.applyConstraints && capabilities.frameRate?.max && native.width && native.height){
            await track.applyConstraints({width:{exact:native.width},height:{exact:native.height},
              frameRate:{ideal:capabilities.frameRate.max},resizeMode:{ideal:'none'}});
          }
        } catch { /* Keep the already negotiated camera mode. */ }
        if(token!==state.session || !root.isConnected){stream.getTracks().forEach(t=>t.stop());return;}
        video.srcObject=stream; await video.play();
        if(token!==state.session || !root.isConnected){stream.getTracks().forEach(t=>t.stop());return;}
        state.active=true;state.seq=0;state.updateQuality();
        root.querySelector('#stop').disabled=false;root.querySelector('#placeholder').hidden=true;
        state.status.textContent='Camera đang chạy · Đang tìm vật…';
        state.component.setStateValue('event',{session:state.session,active:true});
        // Camera enumeration is optional; failure must not tear down a working preview.
        let available=[];
        try { available=await navigator.mediaDevices.enumerateDevices(); } catch { /* Use the default camera. */ }
        if(token!==state.session || !root.isConnected){stream.getTracks().forEach(t=>t.stop());return;}
        const selected=devices.value || stream.getVideoTracks()[0]?.getSettings().deviceId || '';
        devices.replaceChildren(new Option('Camera mặc định',''));
        available.filter(d=>d.kind==='videoinput').forEach((d,i)=>devices.add(new Option(d.label||`Camera ${i+1}`,d.deviceId)));
        devices.value=Array.from(devices.options).some(option=>option.value===selected) ? selected : '';
        stream.getVideoTracks()[0]?.addEventListener('ended',()=>{if(state.stream===stream)state.stop();});
        state.draw(); state.tick();
      } catch(error) {
        if(token!==state.session)return;
        state.stop();
        state.status.textContent=error.name==='NotAllowedError' ? 'Bạn chưa cấp quyền camera. Cho phép camera trong trình duyệt rồi bật lại.' :
          error.name==='NotFoundError' ? 'Không tìm thấy webcam trên thiết bị này.' :
          error.message==='secure' ? 'Camera cần localhost hoặc kết nối HTTPS.' : 'Không mở được camera. Kiểm tra camera có đang được ứng dụng khác sử dụng không.';
      }
    };
    root.querySelector('#start').onclick=start;
    root.querySelector('#stop').onclick=()=>state.stop();
    devices.onchange=()=>{if(state.active)start();};
    overlay.onclick=event=>{
      if(performance.now()-state.frameTime>=2500)return;
      const rect=overlay.getBoundingClientRect(),scale=Math.min(rect.width/state.frameWidth,rect.height/state.frameHeight);
      const x=(event.clientX-rect.left-(rect.width-state.frameWidth*scale)/2)/scale;
      const y=(event.clientY-rect.top-(rect.height-state.frameHeight*scale)/2)/scale;
      const object=state.objects.find(o=>x>=o.box[0]&&x<=o.box[2]&&y>=o.box[1]&&y<=o.box[3]);
      if(object)state.component.setStateValue('selection',{session:state.session,id:object.id,seq:state.seq});
    };
    state.onHide=()=>{state.session=null;state.stop(false);};
    window.addEventListener('pagehide',state.onHide);
    state.observer=new MutationObserver(()=>{if(!root.isConnected){state.session=null;state.stop(false);state.observer.disconnect();window.removeEventListener('pagehide',state.onHide);}});
    state.observer.observe(document,{childList:true,subtree:true});
  }
  state.component=component;state.data=data;
  state.updateQuality();
  const response=data.response;
  if(response && response.session===state.session && response.seq===state.seq && state.busy){
    state.busy=false;state.objects=response.objects||[];state.frameWidth=response.width;state.frameHeight=response.height;
    state.frameTime=performance.now();
    state.status.textContent=response.error || (state.objects.length ? `Đang theo dõi ${state.objects.length} vật · Nhấp khung để xem xác suất.` : 'Chưa phát hiện vật. Đưa vật rõ hơn vào camera.');
  }
  // DOM removal is observed above; cleanup also runs during normal data updates.
  return () => {};
}
