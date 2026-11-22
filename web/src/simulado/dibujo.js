/**
 * Dibuja en un canvas las imágenes de los entornos simulados y las devuelve como JPEG,
 * igual que las enviará el backend. Solo funciona en un navegador (jsdom no tiene canvas):
 * los tests de componentes pasan su propia función de dibujo.
 */

const ANCHO = 600;
const ALTO = 400;

function cartpole(ctx, [x, , theta]) {
  const escala = ANCHO / 4.8;
  const suelo = ALTO * 0.7;
  const cx = ANCHO / 2 + x * escala;
  ctx.fillRect(0, suelo, ANCHO, 2);
  ctx.fillRect(cx - 30, suelo - 20, 60, 30);
  ctx.save();
  ctx.translate(cx, suelo - 20);
  ctx.rotate(theta);
  ctx.fillRect(-5, -escala, 10, escala);
  ctx.restore();
}

function mountaincar(ctx, [posicion]) {
  const aX = (p) => ((p + 1.2) / 1.8) * ANCHO;
  const aY = (p) => ALTO * 0.85 - (Math.sin(3 * p) * 0.45 + 0.55) * ALTO * 0.6;
  ctx.beginPath();
  for (let p = -1.2; p <= 0.6; p += 0.01) {
    ctx.lineTo(aX(p), aY(p));
  }
  ctx.stroke();
  ctx.fillRect(aX(0.5) - 2, aY(0.5) - 40, 4, 40);
  ctx.save();
  ctx.translate(aX(posicion), aY(posicion));
  ctx.rotate(-Math.atan(Math.cos(3 * posicion) * 3 * 0.45 * 0.6 * ALTO / (ANCHO / 1.8)));
  ctx.fillRect(-20, -22, 40, 16);
  ctx.restore();
}

function ejemplo(ctx, [fase], entorno) {
  ctx.font = '20px Arial';
  ctx.fillText(`${entorno} · simulación de ejemplo`, 20, 36);
  ctx.beginPath();
  ctx.arc(ANCHO / 2 + Math.cos(fase) * 140, ALTO / 2 + Math.sin(fase) * 80, 18, 0, 2 * Math.PI);
  ctx.fill();
}

/**
 * Dibuja un fotograma del entorno simulado.
 *
 * @param {string} entorno Clave del entorno.
 * @param {{estado: number[], paso: number}} foto Lo que devuelve Simulacion.foto().
 * @returns {Promise<Blob>} La imagen en JPEG.
 */
export function dibujarFotograma(entorno, { estado }) {
  const lienzo = document.createElement('canvas');
  lienzo.width = ANCHO;
  lienzo.height = ALTO;
  const ctx = lienzo.getContext('2d');
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, ANCHO, ALTO);
  ctx.fillStyle = '#222222';
  ctx.strokeStyle = '#222222';
  ctx.lineWidth = 3;
  if (entorno === 'cartpole') {
    cartpole(ctx, estado);
  } else if (entorno === 'mountaincar') {
    mountaincar(ctx, estado);
  } else {
    ejemplo(ctx, estado, entorno);
  }
  return new Promise((resolver, rechazar) => {
    lienzo.toBlob((imagen) => (imagen ? resolver(imagen) : rechazar(new Error('Sin imagen'))),
      'image/jpeg', 0.8);
  });
}
