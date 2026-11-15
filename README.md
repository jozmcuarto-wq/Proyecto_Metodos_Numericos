# Laboratorio de Métodos Numéricos

Aplicación multipágina para resolver sistemas no lineales y comparar Newton-Raphson (NR), Newton amortiguado (DN) y W4.

## Comparar métodos

La pantalla principal presenta tres sistemas listos para usar: el sistema del informe, un ejemplo polinómico y uno trigonométrico. Elige el punto inicial (P1, P2, P3 o coordenadas propias para el sistema del informe) y compara estado, iteraciones, punto final y residuo en una tabla y una gráfica.

El sistema del informe usa el núcleo numérico validado del proyecto, con Jacobiano analítico. Los ejemplos adicionales usan un Jacobiano aproximado por diferencias finitas y sirven para explorar los métodos; sus resultados no son cifras del informe.

## Análisis del experimento del informe

Las páginas **Cuencas y cobertura**, **Superposición de fallos**, **Trayectorias**, **Fallos y singularidades**, **Sensibilidad**, **Convergencia local** y **Costo** conservan la réplica del sistema

`F1(x,y) = x^2 + y^2 - 4`, `F2(x,y) = x^2*y - 1`.

La configuración de referencia usa una malla inclusiva de 400 × 400 sobre `[-5, 5]^2`, `alpha = 0.5`, `Delta tau = 0.5`, `p0 = (0, 0)`, tolerancia residual `1e-6` y máximo de 1000 iteraciones.

## Ejecutar

Desde la raíz del proyecto:

```powershell
python -m pip install -r app/requirements.txt
python scripts/precompute.py
python -m streamlit run app/app.py
```

La caché `data/baseline_400.npz` contiene estados, iteraciones y distancias singulares para la configuración del informe. Si se regenera, los conteos se validan con:

```powershell
python -m pytest -q app/tests
```

En Streamlit Community Cloud, selecciona `app/app.py` como archivo principal si mantienes esta estructura anidada. Si publicas la app aplanada en la raíz, incluye también `general_solver.py` en la raíz y las versiones actualizadas de `app.py`, `ui.py`, `pages/0_Inicio.py` y `pages/8_Resolver_sistemas.py`.
