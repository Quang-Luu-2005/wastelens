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
    state.stop = (announce=true) => {
      const session=state.session; state.session=null;
      state.active = false;
      state.stream?.getTracks().forEach(track => track.stop());
      state.stream = null; video.srcObject = null; state.busy=false; state.objects=[];
      clearTimeout(state.timer); cancelAnimationFrame(state.animation);
      overlay.getContext('2d').clearRect(0,0,overlay.width,overlay.height);
      root.querySelector('#start').disabled = false; root.querySelector('#stop').disabled = true;
      root.querySelector('#placeholder').hidden = false;
      state.status.textContent = 'Camera đã dừng.';
      if (announce) state.component.setStateValue('event', {session, active:false});
    };
    state.draw = () => {
      if (!state.active) return;
      const width = overlay.clientWidth, height = overlay.clientHeight;
      overlay.width = width; overlay.height = height;
      const ctx = overlay.getContext('2d');
      const scale = Math.min(width/(state.frameWidth||1),height/(state.frameHeight||1));
      const dx=(width-(state.frameWidth||0)*scale)/2, dy=(height-(state.frameHeight||0)*scale)/2;
      if (performance.now()-state.frameTime < 2500) for (const object of state.objects) {
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
        capture.width=Math.min(640,video.videoWidth); capture.height=Math.round(video.videoHeight*capture.width/video.videoWidth);
        // Bound portrait as well as landscape frames.
        if(capture.height>640){capture.width=Math.round(capture.width*640/capture.height);capture.height=640;}
        capture.getContext('2d').drawImage(video,0,0,capture.width,capture.height);
        state.busy=true;state.sentAt=performance.now();state.seq++;
        state.component.setStateValue('frame', {session:state.session,seq:state.seq,
          jpeg:capture.toDataURL('image/jpeg',.8).split(',')[1]});
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
        const stream=await navigator.mediaDevices.getUserMedia({audio:false,video:devices.value ?
          {deviceId:{exact:devices.value},width:{ideal:640},height:{ideal:480}} : {width:{ideal:640},height:{ideal:480},facingMode:'environment'}});
        if(token!==state.session || !root.isConnected){stream.getTracks().forEach(t=>t.stop());return;}
        state.stream=stream; video.srcObject=stream; await video.play(); state.active=true;state.seq=0;
        root.querySelector('#stop').disabled=false;root.querySelector('#placeholder').hidden=true;
        state.status.textContent='Camera đang chạy · Đang tìm vật…';
        state.component.setStateValue('event',{session:state.session,active:true});
        const available=await navigator.mediaDevices.enumerateDevices();
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
  const response=data.response;
  if(response && response.session===state.session && response.seq===state.seq && state.busy){
    state.busy=false;state.objects=response.objects||[];state.frameWidth=response.width;state.frameHeight=response.height;
    state.frameTime=performance.now();
    state.status.textContent=response.error || (state.objects.length ? `Đang theo dõi ${state.objects.length} vật · Nhấp khung để xem xác suất.` : 'Chưa phát hiện vật. Đưa vật rõ hơn vào camera.');
  }
  // DOM removal is observed above; cleanup also runs during normal data updates.
  return () => {};
}
