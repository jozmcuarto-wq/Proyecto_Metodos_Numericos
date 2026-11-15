# Laboratorio de Métodos Numéricos

Aplicación multipágina para resolver sistemas no lineales y comparar Newton-Raphson (NR), Newton amortiguado (DN) y W4.

## Solucionador de sistemas

La página **Resolver sistemas** permite elegir el sistema del proyecto, un ejemplo polinómico, uno trigonométrico o escribir dos funciones propias de `x` e `y`. Los tres métodos parten del mismo punto, comparten la tolerancia residual y muestran estado, iteraciones, residuo, trayectorias, curvas de nivel y archivos CSV descargables.

Se aceptan operaciones `+`, `-`, `*`, `/`, `^` y `**`; funciones `sin`, `cos`, `tan`, `asin`/`arcsin`, `acos`/`arccos`, `atan`/`arctan`, `sinh`, `cosh`, `tanh`, `exp`, `log`, `sqrt`, `abs`; y constantes `pi` y `e`. Las expresiones se interpretan con una lista limitada de operaciones matemáticas; no se ejecuta Python arbitrario.

Para admitir sistemas ingresados por el usuario, esta página aproxima el Jacobiano con diferencias finitas centradas. Sus residuos e iteraciones sirven para comparar los métodos en un mismo problema. Sus tiempos no se comparan con el experimento del informe, cuya implementación usa un Jacobiano analítico y cálculo vectorizado.

## Análisis del experimento del informe

Las páginas **Cuencas y cobertura**, **Superposición de fallos**, **Trayectorias**, **Fallos y singularidades**, **Sensibilidad**, **Convergencia local** y **Costo** conservan la réplica del sistema

`F1(x,y) = x^2 + y^2 - 4`, `F2(x,y) = x^2*y - 1`.

Los parámetros de la malla se comparten en la barra lateral de estas páginas. La configuración de referencia usa una malla inclusiva de 400 × 400 sobre `[-5, 5]^2`, `alpha = 0.5`, `Delta tau = 0.5`, `p0 = (0, 0)`, tolerancia residual `1e-6` y máximo de 1000 iteraciones.

## Ejecutar

Desde la raíz del proyecto:

```powershell
python -m pip install -r app/requirements.txt
python scripts/precompute.py
python -m streamlit run app/app.py
```

La caché `data/baseline_400.npz` contiene los estados, iteraciones y distancias singulares para la configuración del informe. Si se regenera, los conteos se validan con:

```powershell
python -m pytest -q app/tests
```

En Streamlit Community Cloud, selecciona `app/app.py` como archivo principal si mantienes esta estructura anidada. Si publicas la app aplanada en la raíz, incluye también `general_solver.py` en la raíz y la nueva página en `pages/8_Resolver_sistemas.py`, junto con las versiones actualizadas de `app.py`, `ui.py` y `pages/0_Inicio.py`.
