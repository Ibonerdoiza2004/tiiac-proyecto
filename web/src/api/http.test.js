import { describe, expect, it, vi } from 'vitest';

import {
  ErrorApi,
  comprobarRuta,
  construirUrl,
  crearClienteHttp,
  mensajeDeError,
} from './http.js';

/** Respuesta simulada de `fetch`. */
function respuesta(cuerpo, { estado = 200, tipo = 'application/json' } = {}) {
  const texto = cuerpo === undefined ? '' : typeof cuerpo === 'string' ? cuerpo : JSON.stringify(cuerpo);
  return new Response(estado === 204 ? null : texto, {
    status: estado,
    headers: tipo ? { 'content-type': tipo } : {},
  });
}

function clienteCon(...respuestas) {
  const fetch = vi.fn();
  for (const r of respuestas) {
    fetch.mockResolvedValueOnce(r);
  }
  return { fetch, cliente: crearClienteHttp({ fetch }) };
}

describe('comprobarRuta', () => {
  it.each(['/', '/entornos', '/trabajos/3?vista=1'])('acepta %j', (ruta) => {
    expect(comprobarRuta(ruta)).toBe(ruta);
  });

  it.each(['', 'entornos', '//otro-host/x', 'http://x/api', undefined, null, 3, ['/x']])(
    'rechaza %j',
    (ruta) => {
      expect(() => comprobarRuta(ruta)).toThrow(TypeError);
    },
  );
});

describe('construirUrl', () => {
  it.each([
    ['/api', '/entornos', '/api/entornos'],
    ['/api/', '/entornos', '/api/entornos'],
    ['/api//', '/entornos', '/api/entornos'],
    ['', '/entornos', '/entornos'],
    ['http://127.0.0.1:8000/api', '/x', 'http://127.0.0.1:8000/api/x'],
  ])('une %j y %j', (base, ruta, esperada) => {
    expect(construirUrl(base, ruta)).toBe(esperada);
  });

  it('codifica los parámetros y omite los null y undefined', () => {
    const url = construirUrl('/api', '/modelos', {
      entorno: 'cartpole',
      metodo: 'BC y GAIL',
      limite: 0,
      completo: false,
      vacio: null,
      nada: undefined,
    });
    expect(url).toBe('/api/modelos?entorno=cartpole&metodo=BC+y+GAIL&limite=0&completo=false');
  });

  it('añade los parámetros a una ruta que ya tiene consulta', () => {
    expect(construirUrl('/api', '/x?a=1', { b: 2 })).toBe('/api/x?a=1&b=2');
  });

  it('sin parámetros no añade «?»', () => {
    expect(construirUrl('/api', '/x', {})).toBe('/api/x');
  });
});

describe('mensajeDeError', () => {
  it.each([
    [{ detail: 'Modelo no encontrado' }, 404, 'Modelo no encontrado'],
    [
      { detail: [{ msg: 'Field required' }, { msg: 'Input should be a valid integer' }] },
      422,
      'Field required; Input should be a valid integer',
    ],
    [{ detail: [{ loc: ['body'] }] }, 422, 'La API respondió con el código 422'],
    [{ detail: '   ' }, 500, 'La API respondió con el código 500'],
    [{ otro: 'x' }, 500, 'La API respondió con el código 500'],
    ['Internal Server Error', 500, 'Internal Server Error'],
    [null, 502, 'La API respondió con el código 502'],
  ])('%j con %i da %j', (cuerpo, estado, esperado) => {
    expect(mensajeDeError(cuerpo, estado)).toBe(esperado);
  });
});

describe('crearClienteHttp', () => {
  it('GET devuelve el JSON y pide JSON', async () => {
    const { fetch, cliente } = clienteCon(respuesta([{ id: 'cartpole' }]));
    await expect(cliente.obtener('/entornos')).resolves.toEqual([{ id: 'cartpole' }]);
    const [url, init] = fetch.mock.calls[0];
    expect(url).toBe('/api/entornos');
    expect(init.method).toBe('GET');
    expect(init.headers).toEqual({ Accept: 'application/json' });
    expect(init.body).toBeUndefined();
  });

  it.each([
    ['enviar', 'POST'],
    ['actualizar', 'PATCH'],
  ])('%s manda el cuerpo como JSON con %s', async (metodo, verbo) => {
    const { fetch, cliente } = clienteCon(respuesta({ id: 7 }, { estado: 201 }));
    const cuerpo = { metodo: 'GAIL', cantidad: 4, ajustes: { gail: { iteraciones: 20 } } };
    await expect(cliente[metodo]('/trabajos', cuerpo)).resolves.toEqual({ id: 7 });
    const [, init] = fetch.mock.calls[0];
    expect(init.method).toBe(verbo);
    expect(init.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(init.body)).toEqual(cuerpo);
  });

  it('enviar sin cuerpo no manda Content-Type', async () => {
    const { fetch, cliente } = clienteCon(respuesta(undefined, { estado: 204 }));
    await expect(cliente.enviar('/trabajos/3/pausar')).resolves.toBeNull();
    expect(fetch.mock.calls[0][1].headers).not.toHaveProperty('Content-Type');
  });

  it.each([false, 0, null, ''])('enviar manda también el cuerpo %j', async (cuerpo) => {
    const { fetch, cliente } = clienteCon(respuesta({}));
    await cliente.enviar('/x', cuerpo);
    expect(fetch.mock.calls[0][1].body).toBe(JSON.stringify(cuerpo));
  });

  it('DELETE con 204 devuelve null', async () => {
    const { fetch, cliente } = clienteCon(respuesta(undefined, { estado: 204 }));
    await expect(cliente.borrar('/trabajos/3')).resolves.toBeNull();
    expect(fetch.mock.calls[0][1].method).toBe('DELETE');
  });

  it('un cuerpo vacío con 200 da null', async () => {
    const { cliente } = clienteCon(respuesta('', { tipo: 'application/json' }));
    await expect(cliente.obtener('/x')).resolves.toBeNull();
  });

  it('un cuerpo de texto se devuelve como texto', async () => {
    const { cliente } = clienteCon(respuesta('hola', { tipo: 'text/plain' }));
    await expect(cliente.obtener('/x')).resolves.toBe('hola');
  });

  it('un JSON roto con 200 es un error', async () => {
    const { cliente } = clienteCon(respuesta('{roto', { tipo: 'application/json' }));
    const error = await cliente.obtener('/x').catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(200);
    expect(error.detalle).toBe('{roto');
    expect(error.cause).toBeInstanceOf(SyntaxError);
  });

  it.each([
    [404, { detail: 'Modelo no encontrado' }, 'application/json', 'Modelo no encontrado'],
    [422, { detail: [{ msg: 'Field required' }] }, 'application/json', 'Field required'],
    [500, 'Internal Server Error', 'text/plain', 'Internal Server Error'],
    [503, '{roto', 'application/json', '{roto'],
    [500, undefined, null, 'La API respondió con el código 500'],
  ])('un %i da ErrorApi con su mensaje', async (estado, cuerpo, tipo, mensaje) => {
    const { cliente } = clienteCon(respuesta(cuerpo, { estado, tipo }));
    const error = await cliente.obtener('/modelos/9').catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(estado);
    expect(error.message).toBe(mensaje);
  });

  it('guarda el cuerpo de error en detalle', async () => {
    const cuerpo = { detail: [{ msg: 'Field required', loc: ['body', 'metodo'] }] };
    const { cliente } = clienteCon(respuesta(cuerpo, { estado: 422 }));
    const error = await cliente.enviar('/trabajos', {}).catch((e) => e);
    expect(error.detalle).toEqual(cuerpo);
  });

  it('un fallo de red da ErrorApi con estado 0 y la causa', async () => {
    const fallo = new TypeError('Failed to fetch');
    const fetch = vi.fn().mockRejectedValue(fallo);
    const error = await crearClienteHttp({ fetch }).obtener('/x').catch((e) => e);
    expect(error).toBeInstanceOf(ErrorApi);
    expect(error.estado).toBe(0);
    expect(error.message).toBe('No se pudo conectar con la API');
    expect(error.cause).toBe(fallo);
  });

  it('una petición cancelada propaga el AbortError tal cual', async () => {
    const control = new AbortController();
    const fetch = vi.fn((_, { signal }) =>
      new Promise((_, rechazar) => {
        signal.addEventListener('abort', () => rechazar(new DOMException('cancelada', 'AbortError')));
      }),
    );
    const promesa = crearClienteHttp({ fetch }).obtener('/x', { senal: control.signal });
    control.abort();
    const error = await promesa.catch((e) => e);
    expect(error.name).toBe('AbortError');
    expect(error).not.toBeInstanceOf(ErrorApi);
  });

  it('pasa los parámetros de consulta', async () => {
    const { fetch, cliente } = clienteCon(respuesta([]));
    await cliente.obtener('/modelos', { parametros: { entorno: 'acrobot' } });
    expect(fetch.mock.calls[0][0]).toBe('/api/modelos?entorno=acrobot');
  });

  it('usa la base indicada', async () => {
    const fetch = vi.fn().mockResolvedValue(respuesta({}));
    await crearClienteHttp({ base: 'http://127.0.0.1:8000/api/', fetch }).obtener('/salud');
    expect(fetch.mock.calls[0][0]).toBe('http://127.0.0.1:8000/api/salud');
  });

  it('sin fetch propio usa el global en el momento de la llamada', async () => {
    const fetch = vi.fn().mockResolvedValue(respuesta({ ok: true }));
    const cliente = crearClienteHttp();
    vi.stubGlobal('fetch', fetch);
    try {
      await expect(cliente.obtener('/x')).resolves.toEqual({ ok: true });
    } finally {
      vi.unstubAllGlobals();
    }
    expect(fetch).toHaveBeenCalledOnce();
  });

  it.each(['entornos', '', '//x'])('rechaza la ruta %j sin llamar a fetch', async (ruta) => {
    const { fetch, cliente } = clienteCon();
    await expect(cliente.obtener(ruta)).rejects.toThrow(TypeError);
    expect(fetch).not.toHaveBeenCalled();
  });
});

describe('ErrorApi', () => {
  it('sin datos tiene estado 0 y detalle null', () => {
    const error = new ErrorApi('x');
    expect(error).toBeInstanceOf(Error);
    expect(error.name).toBe('ErrorApi');
    expect(error.estado).toBe(0);
    expect(error.detalle).toBeNull();
    expect(error.cause).toBeUndefined();
  });
});
