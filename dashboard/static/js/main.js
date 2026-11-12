// ── slider live values ──────────────────────────────────────────────────────
document.querySelectorAll('input[type=range]').forEach(el => {
  const out = el.nextElementSibling;
  if (out && out.tagName === 'SPAN') out.textContent = el.value;
  el.addEventListener('input', () => {
    const o = el.nextElementSibling;
    if (o && o.tagName === 'SPAN') o.textContent = el.value;
  });
});

// ── gauge SVG ───────────────────────────────────────────────────────────────
function drawGauge(value, max = 10) {
  const wrap = document.getElementById('gauge-wrap');
  if (!wrap) return;
  const pct   = Math.min(value / max, 1);
  const angle = pct * 180 - 90;
  const r     = 90, cx = 120, cy = 110;
  const toRad = d => d * Math.PI / 180;
  const arc   = (a1, a2, col) => {
    const x1 = cx + r * Math.cos(toRad(a1 - 90));
    const y1 = cy + r * Math.sin(toRad(a1 - 90));
    const x2 = cx + r * Math.cos(toRad(a2 - 90));
    const y2 = cy + r * Math.sin(toRad(a2 - 90));
    const lg = (a2 - a1) > 180 ? 1 : 0;
    return `<path d="M${x1},${y1} A${r},${r} 0 ${lg},1 ${x2},${y2}" fill="none" stroke="${col}" stroke-width="14" stroke-linecap="round"/>`;
  };
  const needle = () => {
    const a  = angle;
    const nx = cx + (r - 10) * Math.cos(toRad(a));
    const ny = cy + (r - 10) * Math.sin(toRad(a));
    return `<line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#4ade80" stroke-width="3" stroke-linecap="round"/>
            <circle cx="${cx}" cy="${cy}" r="6" fill="#4ade80"/>`;
  };
  const color = pct < 0.35 ? '#ef4444' : pct < 0.65 ? '#fbbf24' : '#4ade80';
  wrap.innerHTML = `
    <svg width="240" height="140" viewBox="0 0 240 140">
      ${arc(-90, 90, 'rgba(255,255,255,0.06)')}
      ${arc(-90, -90 + pct * 180, color)}
      ${needle()}
      <text x="${cx}" y="${cy + 28}" text-anchor="middle" fill="${color}" font-size="22" font-weight="800">${value}</text>
      <text x="${cx}" y="${cy + 44}" text-anchor="middle" fill="#6b7280" font-size="11">t/ha</text>

    </svg>`;
}

// ── EDA charts ──────────────────────────────────────────────────────────────
function initEDA(raw) {
  const d = JSON.parse(raw);
  if (!d || !d.weeks) return;

  // NDVI trend
  const ndviCtx = document.getElementById('ndviChart');
  if (ndviCtx) {
    const upper = d.ndvi_mean.map((v, i) => v + d.ndvi_std[i]);
    const lower = d.ndvi_mean.map((v, i) => v - d.ndvi_std[i]);
    new Chart(ndviCtx, {
      type: 'line',
      data: {
        labels: d.weeks,
        datasets: [
          { label: '+1 std', data: upper, borderColor: 'transparent',
            backgroundColor: 'rgba(74,222,128,0.15)', pointRadius: 0, tension: 0.4, fill: '+1' },
          { label: 'Mean NDVI', data: d.ndvi_mean, borderColor: '#4ade80', borderWidth: 2.5,
            pointRadius: 3, pointBackgroundColor: '#4ade80', tension: 0.4, fill: false },
          { label: '-1 std', data: lower, borderColor: 'transparent',
            backgroundColor: 'rgba(74,222,128,0.15)', pointRadius: 0, tension: 0.4, fill: '-1' },
        ]
      },
      options: chartOpts('Week (Growing Season)', 'NDVI')
    });
  }

  // Yield bar chart — all 6 crops with real avg values
  const yieldCtx = document.getElementById('yieldChart');
  if (yieldCtx && (d.crop_means || d.crop_yields)) {
    let crops, avgs;
    if (d.crop_means) {
      crops = Object.keys(d.crop_means);
      avgs  = Object.values(d.crop_means);
    } else {
      crops = Object.keys(d.crop_yields);
      avgs  = crops.map(c => +(d.crop_yields[c].reduce((a,b)=>a+b,0)/d.crop_yields[c].length).toFixed(3));
    }
    const colors = ['#4ade80','#60a5fa','#fbbf24','#f87171','#a78bfa','#fb923c'];
    new Chart(yieldCtx, {
      type: 'bar',
      data: {
        labels: crops,
        datasets: [{
          label: 'Avg Yield (t/ha)',
          data: avgs,
          backgroundColor: colors.slice(0, crops.length).map(c => c + '88'),
          borderColor: colors.slice(0, crops.length),
          borderWidth: 2,
          borderRadius: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: { backgroundColor: '#1f2937', titleColor: '#f0fdf4', bodyColor: '#9ca3af',
                     borderColor: '#374151', borderWidth: 1 }
        },
        scales: {
          x: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' } },
          y: { ticks: { color: '#9ca3af' }, grid: { color: 'rgba(255,255,255,0.04)' },
               title: { display: true, text: 'Avg Yield (t/ha)', color: '#6b7280' } }
        }
      }
    });
  }

  // Correlation heatmap (canvas-based)
  const corrCtx = document.getElementById('corrChart');
  if (corrCtx && d.corr_labels) {
    drawHeatmap(corrCtx, d.corr_labels, d.corr_matrix);
  }
}

function drawHeatmap(canvas, labels, matrix) {
  const n    = labels.length;
  const par  = canvas.parentElement;
  const size = Math.min(par.offsetWidth || 400, 420);
  canvas.width  = size;
  canvas.height = size;
  const ctx  = canvas.getContext('2d');
  const off  = 90;
  const cell = (size - off - 10) / n;

  ctx.clearRect(0, 0, size, size);

  labels.forEach((l, i) => {
    ctx.fillStyle = '#9ca3af';
    ctx.font = '10px Inter';
    ctx.textAlign = 'right';
    ctx.fillText(l, off - 6, off + i * cell + cell / 2 + 4);
    ctx.textAlign = 'center';
    ctx.save();
    ctx.translate(off + i * cell + cell / 2, off - 10);
    ctx.rotate(-Math.PI / 4);
    ctx.fillText(l, 0, 0);
    ctx.restore();
  });

  matrix.forEach((row, i) => {
    row.forEach((val, j) => {
      const v = Math.max(-1, Math.min(1, val));
      const alpha = Math.abs(v) * 0.85 + 0.1;
      const color = v > 0 ? `rgba(74,222,128,${alpha})` : `rgba(239,68,68,${alpha})`;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(off + j * cell + 2, off + i * cell + 2, cell - 4, cell - 4, 4);
      ctx.fill();
      ctx.fillStyle = Math.abs(v) > 0.5 ? '#000' : '#fff';
      ctx.font = `bold ${Math.max(9, Math.floor(cell * 0.28))}px Inter`;
      ctx.textAlign = 'center';
      ctx.fillText(val.toFixed(2), off + j * cell + cell / 2, off + i * cell + cell / 2 + 4);
    });
  });
}

// ── comparison charts ────────────────────────────────────────────────────────
function initComparison(raw) {
  const d = JSON.parse(raw);
  if (!d || !d.models) return;

  const rmseCtx = document.getElementById('rmseChart');
  if (rmseCtx) {
    new Chart(rmseCtx, {
      type: 'bar',
      data: {
        labels: d.models,
        datasets: [{ label: 'RMSE', data: d.rmse,
          backgroundColor: d.rmse.map(v => v === Math.min(...d.rmse) ? '#4ade80aa' : '#ffffff18'),
          borderColor:     d.rmse.map(v => v === Math.min(...d.rmse) ? '#4ade80'   : '#ffffff33'),
          borderWidth: 2, borderRadius: 8 }]
      },
      options: { ...chartOpts('Model', 'RMSE'), plugins: { legend: { display: false } } }
    });
  }

  const r2Ctx = document.getElementById('r2Chart');
  if (r2Ctx) {
    new Chart(r2Ctx, {
      type: 'bar',
      data: {
        labels: d.models,
        datasets: [{ label: 'R²', data: d.r2,
          backgroundColor: d.r2.map(v => v === Math.max(...d.r2) ? '#60a5faaa' : '#ffffff18'),
          borderColor:     d.r2.map(v => v === Math.max(...d.r2) ? '#60a5fa'   : '#ffffff33'),
          borderWidth: 2, borderRadius: 8 }]
      },
      options: { ...chartOpts('Model', 'R²'), plugins: { legend: { display: false } } }
    });
  }
}

// ── shared chart defaults ────────────────────────────────────────────────────
function chartOpts(xLabel, yLabel) {
  return {
    responsive: true, maintainAspectRatio: false,
    plugins: {
      legend: { labels: { color: '#9ca3af', font: { family: 'Inter', size: 12 } } },
      tooltip: { backgroundColor: '#1f2937', titleColor: '#f0fdf4', bodyColor: '#9ca3af',
                 borderColor: '#374151', borderWidth: 1 }
    },
    scales: {
      x: { ticks: { color: '#6b7280', font: { family: 'Inter' } },
           grid: { color: 'rgba(255,255,255,0.04)' },
           title: { display: true, text: xLabel, color: '#6b7280' } },
      y: { ticks: { color: '#6b7280', font: { family: 'Inter' } },
           grid: { color: 'rgba(255,255,255,0.04)' },
           title: { display: true, text: yLabel, color: '#6b7280' } }
    }
  };
}

// ── init on load ─────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // gauge removed

  // EDA
  const edaData = document.getElementById('eda-data');
  if (edaData) initEDA(edaData.dataset.json);

  // Comparison
  const cmpData = document.getElementById('cmp-data');
  if (cmpData) initComparison(cmpData.dataset.json);
});
