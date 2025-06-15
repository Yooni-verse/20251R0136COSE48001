const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

ctx.fillStyle = "#ffffff";
ctx.fillRect(0, 0, canvas.width, canvas.height);

let drawing = false;
canvas.addEventListener('pointerdown', e => {
  drawing = true;
  ctx.lineWidth = document.getElementById("pen-size").value;
  ctx.lineCap = 'round';
  const penType = document.getElementById("pen-type").value;
  ctx.strokeStyle = penType === 'eraser' ? "#ffffff" : "#000000";

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

  // Top 1 이미지
  const topImg = document.createElement("img");
  topImg.src = results[0].image;
  topImg.className = "top-result";
  resultEl.appendChild(topImg);

  // ✅ Top 이미지 클릭 가능하게
  topImg.onclick = () => {
    const item = results[0];
    document.getElementById("detail-image").src = item.image;
    document.getElementById("item-name").textContent = item.name;
    document.getElementById("item-price").textContent = `₩${Number(item.price).toLocaleString()}`;
    document.getElementById("item-link").href = item.link;
    document.getElementById("item-link").textContent = "Buy now";
    document.getElementById("item-description").textContent = item.description;
    document.getElementById("item-rating").textContent = `⭐ ${item.rating} (${item.num_reviews} reviews)`;

    const detailView = document.getElementById("detail-view");
    detailView.classList.remove("hidden");
    void detailView.offsetWidth;
    detailView.classList.add("visible");
    detailView.scrollIntoView({ behavior: "smooth" });
  };

  const thumbContainer = document.createElement("div");
  thumbContainer.className = "thumb-container";

  results.forEach((item, idx) => {
    if (idx === 0) return;

    const img = document.createElement("img");
    img.src = item.image;
    img.className = "sub-result";

    img.onclick = () => {
      document.getElementById("detail-image").src = item.image;
      document.getElementById("item-name").textContent = item.name;
      document.getElementById("item-price").textContent = `₩${Number(item.price).toLocaleString()}`;
      document.getElementById("item-link").href = item.link;
      document.getElementById("item-link").textContent = "Buy now";
      document.getElementById("item-description").textContent = item.description;
      document.getElementById("item-rating").textContent = `⭐ ${item.rating} (${item.num_reviews} reviews)`;

      const detailView = document.getElementById("detail-view");
      detailView.classList.remove("hidden");
      void detailView.offsetWidth;
      detailView.classList.add("visible");
      detailView.scrollIntoView({ behavior: "smooth" });
    };

    thumbContainer.appendChild(img);
  });

  resultEl.appendChild(thumbContainer);
  resultEl.classList.remove("hidden");
  void resultEl.offsetWidth;
  resultEl.classList.add("visible");

  document.querySelector(".interaction-container").classList.add("show-result");
};
