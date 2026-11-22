import { useEffect, useState } from 'react';

/**
 * Muestra la última imagen del entorno que ha llegado por el WebSocket.
 *
 * @param {object} props
 * @param {Blob | null} props.imagen Última imagen (JPEG), o null si todavía no hay.
 * @param {string} props.nombre Nombre del entorno, para el texto alternativo.
 * @param {string} props.estado Etiqueta del estado (por ejemplo «En marcha»).
 * @param {string} [props.aviso] Texto sobre la imagen, como la ayuda del teclado.
 * @returns {JSX.Element}
 */
export default function VistaEntorno({ imagen, nombre, estado, aviso }) {
  const [url, setUrl] = useState(null);

  useEffect(() => {
    if (!imagen) {
      setUrl(null);
      return undefined;
    }
    const nueva = URL.createObjectURL(imagen);
    setUrl(nueva);
    return () => URL.revokeObjectURL(nueva);
  }, [imagen]);

  return (
    <div className="escena">
      <span className="escena-etiqueta">Vista del entorno</span>
      <span className="escena-estado" role="status">{estado}</span>
      {url ? <img src={url} alt={`${nombre}: última imagen de la partida`} />
        : <p className="escena-vacia">Inicia una sesión para ver el entorno.</p>}
      {aviso ? <p className="escena-aviso">{aviso}</p> : null}
    </div>
  );
}
