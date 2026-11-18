// All code/assets are sent inline with the WebSocket-rendered component. No per-instance HTTP files.
export default function(component) {
  const { data, parentElement, setStateValue } = component;
  const upload = parentElement.querySelector("#upload");
  const file = parentElement.querySelector("#file");
  const downloads = parentElement.querySelector("#downloads");
  const error = parentElement.querySelector("#error");
  const urls = [];
  upload.hidden = data.mode !== "upload";
  downloads.hidden = data.mode !== "downloads";
  parentElement.querySelector("#pick").onclick = () => file.click();

  function sendFile(selected) {
    if (!selected) return;
    error.textContent = "";
    if (selected.size > data.max_bytes) {
      error.textContent = "Ảnh lớn hơn 4 MB. Vui lòng chọn ảnh nhỏ hơn.";
      return;
    }
    if (!/\.(jpe?g|png)$/i.test(selected.name)) {
      error.textContent = "Vui lòng chọn ảnh JPG hoặc PNG.";
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setStateValue("file", { name: selected.name, data: String(reader.result).split(",")[1] });
      file.value = "";
    };
    reader.onerror = () => { error.textContent = "Không đọc được ảnh. Vui lòng chọn lại."; };
    reader.readAsDataURL(selected);
  }

  file.onchange = () => sendFile(file.files[0]);
  upload.ondragover = event => { event.preventDefault(); upload.classList.add("drag"); };
  upload.ondragleave = () => upload.classList.remove("drag");
  upload.ondrop = event => {
    event.preventDefault(); upload.classList.remove("drag"); sendFile(event.dataTransfer.files[0]);
  };
  if (data.mode === "downloads") {
    downloads.replaceChildren();
    for (const item of data.files) {
      const bytes = Uint8Array.from(atob(item.data), char => char.charCodeAt(0));
      const url = URL.createObjectURL(new Blob([bytes], { type: item.mime }));
      urls.push(url);
      const link = document.createElement("a");
      link.href = url; link.download = item.name; link.textContent = item.label;
      downloads.append(link);
    }
  }
  return () => urls.forEach(url => URL.revokeObjectURL(url));
}
