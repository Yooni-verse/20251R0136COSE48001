const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;

canvas.addEventListener('pointerdown', e => {
  drawing = true;

  const penType = document.getElementById("pen-type").value;
  const penSize = parseInt(document.getElementById("pen-size").value, 3);

  ctx.lineWidth = penSize;
  ctx.lineCap = 'round';

  if (penType === "pencil") ctx.strokeStyle = "#666666";
  else if (penType === "pen") ctx.strokeStyle = "#000000";
  else if (penType === "eraser") ctx.strokeStyle = "#ffffff";

  const r = canvas.getBoundingClientRect();
  ctx.beginPath();
  ctx.moveTo(e.clientX - r.left, e.clientY - r.top);
});

canvas.addEventListener('pointerup', () => {
  drawing = false;
  ctx.beginPath();
});

canvas.addEventListener('pointermove', e => {
  if (!drawing) return;
  const r = canvas.getBoundingClientRect();
  ctx.lineTo(e.clientX - r.left, e.clientY - r.top);
  ctx.stroke();
});

function resetCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.beginPath();
}

document.getElementById("clear").onclick = resetCanvas;

document.getElementById("submit").onclick = async () => {
  const caption = document.getElementById("caption").value;
  const dataURL = canvas.toDataURL("image/png");

  const res = await fetch("/infer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sketch: dataURL, caption })
  });

  if (!res.ok) {
    alert("Error " + res.status);
    return;
  }

  const { results } = await res.json();
  const resultEl = document.getElementById("result");
  resultEl.innerHTML = "";

  const topImg = document.createElement("img");
  topImg.src = results[0];
  topImg.className = "top-result";
  resultEl.appendChild(topImg);

  topImg.onclick = () => {
    showDetail(results[0], 1);
  };

  const thumbContainer = document.createElement("div");
  thumbContainer.className = "thumb-container";

  results.slice(1).forEach((b64, idx) => {
    const img = document.createElement("img");
    img.src = b64;
    img.className = "sub-result";
    img.onclick = () => {
      showDetail(b64, idx + 2);
    };
    thumbContainer.appendChild(img);
  });

  resultEl.appendChild(thumbContainer);

  resultEl.classList.remove("hidden");
  void resultEl.offsetWidth;
  resultEl.classList.add("visible");

  document.querySelector(".interaction-container").classList.add("show-result");
};

function showDetail(imageSrc, index) {
  document.getElementById("detail-image").src = imageSrc;
  document.getElementById("item-name").textContent = `Item #${index}`;
  document.getElementById("item-price").textContent = `Price: $${index * 10}.00`;
  document.getElementById("item-link").href = "#";
  document.getElementById("item-link").textContent = "Buy now";

  const detailView = document.getElementById("detail-view");
  detailView.classList.remove("hidden");
  void detailView.offsetWidth;
  detailView.classList.add("visible");
  detailView.scrollIntoView({ behavior: "smooth", block: "start" });
}

// 스크롤 위로
const scrollBtn = document.getElementById("scroll-top-btn");

window.addEventListener("scroll", () => {
  if (window.scrollY > 300) scrollBtn.classList.add("show");
  else scrollBtn.classList.remove("show");
});

scrollBtn.onclick = () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
};
