import { describe, expect, it } from 'vitest';

import { aFormulario, camposPorRuta, hojasDeAjustes, idCampo, leerCampo, validarAjustes,
  valoresIniciales } from './ajustes.js';

const ENTERO = { clave: 'n', tipo: 'entero', min: 1, valor: 20 };
const ENTERO_LIBRE = { clave: 'r', tipo: 'entero', valor: -90 };
const REAL = { clave: 'lr', tipo: 'real', valor: 0.01 };
const REAL_MIN = { clave: 'g', tipo: 'real', min: 0, valor: 0.5 };
const BOOLEANO = { clave: 'b', tipo: 'booleano', valor: true };
const OPCION = { clave: 'activacion', tipo: 'opcion', opciones: ['relu', 'tanh'], valor: 'tanh' };
const LISTA = { clave: 'capas', tipo: 'lista_enteros', min: 1, valor: [100, 100] };

const ESQUEMA = { metodo: 'gail', duracion: { unidad: 'iteraciones', total: 20 }, secciones: [
  { clave: 'gail', campos: [ENTERO, BOOLEANO] },
  { clave: 'gail.discriminador', campos: [REAL, OPCION, LISTA] },
  { clave: 'experto.trpo', campos: [ENTERO_LIBRE] },
] };

describe('leerCampo', () => {
  it.each([
    [ENTERO, '20', 20], [ENTERO, ' 7 ', 7], [ENTERO, '+3', 3], [ENTERO, '1', 1],
    [ENTERO_LIBRE, '-90', -90], [ENTERO_LIBRE, '0', 0],
    [REAL, '0.02', 0.02], [REAL, '0,02', 0.02], [REAL, '1e-3', 0.001], [REAL, '1E3', 1000],
    [REAL, '.5', 0.5], [REAL, '5.', 5], [REAL, '-3.5', -3.5], [REAL, '7', 7], [REAL_MIN, '0', 0],
    [BOOLEANO, false, false], [BOOLEANO, true, true],
    [OPCION, 'relu', 'relu'],
    [LISTA, '64, 64', [64, 64]], [LISTA, '256', [256]], [LISTA, '32 64;128', [32, 64, 128]],
  ])('%j acepta %j', (campo, entrada, valor) => {
    expect(leerCampo(campo, entrada)).toEqual({ valor });
  });

  it.each([
    [ENTERO, '', 'entero'], [ENTERO, '2.5', 'entero'], [ENTERO, 'diez', 'entero'], [ENTERO, '1e3', 'entero'],
    [ENTERO, '0', 'mínimo 1'], [ENTERO, '-1', 'mínimo 1'], [ENTERO, '9007199254740993', 'entero'],
    [REAL, '', 'número'], [REAL, 'abc', 'número'], [REAL, '1e999', 'número'], [REAL, 'NaN', 'número'],
    [REAL, 'Infinity', 'número'], [REAL, '0,0,1', 'número'], [REAL, '1.2.3', 'número'],
    [REAL_MIN, '-0.1', 'mínimo 0'],
    [BOOLEANO, 'true', 'sí o no'],
    [OPCION, 'sigmoid', 'relu, tanh'], [OPCION, '', 'relu, tanh'],
    [LISTA, '', 'separados por comas'], [LISTA, '64, x', 'separados por comas'], [LISTA, '1.5', 'separados'],
    [LISTA, '64, 0', 'mínimo 1'],
    [{ clave: 'x', tipo: 'fecha', valor: 1 }, '1', 'desconocido'],
  ])('%j rechaza %j', (campo, entrada, mensaje) => {
    const leido = leerCampo(campo, entrada);
    expect(leido).toHaveProperty('error');
    expect(leido.error).toContain(mensaje);
  });
});

describe('formulario', () => {
  it('parte de los valores del perfil, como texto salvo las casillas', () => {
    expect(valoresIniciales(ESQUEMA)).toEqual({
      'gail.n': '20', 'gail.b': true, 'gail.discriminador.lr': '0.01',
      'gail.discriminador.activacion': 'tanh', 'gail.discriminador.capas': '100, 100',
      'experto.trpo.r': '-90',
    });
    expect(aFormulario(REAL, 1e-6)).toBe('0.000001');
    expect(idCampo({ clave: 'a.b' }, { clave: 'c' })).toBe('a.b.c');
  });

  it('sin cambios no envía ajustes', () => {
    expect(validarAjustes(ESQUEMA, valoresIniciales(ESQUEMA)))
      .toEqual({ valido: true, errores: {}, ajustes: {}, cambiados: 0 });
    // Escribir lo mismo de otra forma tampoco es un cambio.
    const otraForma = { ...valoresIniciales(ESQUEMA), 'gail.discriminador.lr': '1e-2',
      'gail.discriminador.capas': '100,100', 'gail.n': ' 20 ' };
    expect(validarAjustes(ESQUEMA, otraForma).cambiados).toBe(0);
  });

  it('envía solo lo cambiado, anidado como en configs/', () => {
    const valores = { ...valoresIniciales(ESQUEMA), 'gail.n': '50', 'gail.b': false,
      'gail.discriminador.capas': '64, 64', 'experto.trpo.r': '-80' };
    expect(validarAjustes(ESQUEMA, valores)).toEqual({ valido: true, errores: {}, cambiados: 4,
      ajustes: { gail: { n: 50, b: false, discriminador: { capas: [64, 64] } }, experto: { trpo: { r: -80 } } } });
  });

  it('marca cada campo que no vale y no envía nada de él', () => {
    const valores = { ...valoresIniciales(ESQUEMA), 'gail.n': '0', 'gail.discriminador.lr': 'x',
      'gail.discriminador.activacion': 'tanh' };
    const resultado = validarAjustes(ESQUEMA, valores);
    expect(resultado.valido).toBe(false);
    expect(Object.keys(resultado.errores).sort()).toEqual(['gail.discriminador.lr', 'gail.n']);
    expect(resultado.ajustes).toEqual({});
  });

  it('un campo que falta en el formulario vale lo del perfil', () => {
    expect(validarAjustes(ESQUEMA, {})).toMatchObject({ valido: true, cambiados: 0 });
  });
});

describe('ajustes ya construidos', () => {
  it('se recorren por ruta', () => {
    expect(hojasDeAjustes({ gail: { n: 5, discriminador: { capas: [1, 2] } }, bc: {} }))
      .toEqual([['gail.n', 5], ['gail.discriminador.capas', [1, 2]]]);
    expect(hojasDeAjustes({})).toEqual([]);
  });

  it.each([null, [], 'texto', 3])('rechazan %j como ajustes', (ajustes) => {
    expect(() => hojasDeAjustes(ajustes)).toThrow(TypeError);
  });

  it('cada ruta tiene su campo', () => {
    const campos = camposPorRuta(ESQUEMA);
    expect([...campos.keys()]).toEqual(['gail.n', 'gail.b', 'gail.discriminador.lr',
      'gail.discriminador.activacion', 'gail.discriminador.capas', 'experto.trpo.r']);
    expect(campos.get('gail.discriminador.lr')).toBe(REAL);
  });
});
