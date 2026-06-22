# Arquitectura Matemática de la Physics-Informed Neural Network (PINN) para Tomografía de Resistividad Eléctrica (ERT) 3D

Este documento formaliza la formulación analítica, el mapeo de redes neuronales, las funciones de pérdida física y las condiciones de frontera que gobiernan la inversión de ERT tridimensional.

---

## 1. Mapeo de Redes y Restricciones de Continuidad

El problema inverso busca reconstruir el campo de conductividad 3D subsuperficial a partir de mediciones de potencial eléctrico en la superficie. Para ello, instanciamos dos redes neuronales acopladas:

1. **Red de Conductividad $\sigma_\theta$**:
   Mapea coordenadas espaciales al valor de conductividad local.
   $$ \sigma_\theta: \mathbb{R}^3 \to \mathbb{R}^+ $$
   $$ \sigma(x, y, z) = f_\theta(x, y, z) $$
   *(Nota: Se aplicará una función softplus u operación exponencial en la última capa para garantizar $\sigma > 0$)*.

2. **Red de Potencial Eléctrico $u_\phi$**:
   Mapea coordenadas espaciales al potencial eléctrico inducido por un dipolo de corriente. Dado que el potencial depende de la posición del dipolo inyector $(A, B)$, la red toma también los índices o coordenadas de la fuente:
   $$ u_\phi: \mathbb{R}^3 \times \mathcal{S} \to \mathbb{R} $$
   $$ u(x, y, z | \mathbf{r}_A, \mathbf{r}_B) = g_\phi(x, y, z, \mathbf{r}_A, \mathbf{r}_B) $$

> [!WARNING]
> **Continuidad Diferencial**: Dado que el residual físico evalúa la Ecuación de Poisson (la cual involucra segundas derivadas $\nabla^2 u$ y primeras derivadas $\nabla \sigma$), **el uso de activaciones tipo ReLU está estrictamente prohibido**. ReLU tiene una segunda derivada nula en casi todo su dominio, lo que haría colapsar el Hessiano a cero (vanishing gradient sobre la física).
> **Todas las capas ocultas usarán activaciones $C^2$-continuas (SiLU / Swish, Tanh o GELU)**.

---

## 2. Componentes del Funcional de Pérdida ($\mathcal{L}_{Total}$)

La PINN se entrenará minimizando un funcional multiobjetivo compuesto:
$$ \mathcal{L}_{Total}(\theta, \phi) = \lambda_{data} \mathcal{L}_{data} + \lambda_{PDE} \mathcal{L}_{PDE} + \lambda_{bc} \mathcal{L}_{BC} + \lambda_{reg} \mathcal{L}_{reg} + \lambda_{flux} \mathcal{L}_{flux} $$

### 2.1. Función de Pérdida de Datos ($\mathcal{L}_{data}$)

Garantiza que el potencial eléctrico predicho por la red en la superficie coincida con las mediciones empíricas almacenadas en el dataset HDF5.

$$ \mathcal{L}_{data} = \frac{1}{N_{meas}} \sum_{k=1}^{N_{meas}} || u_\phi(x_{M,k}, y_{M,k}, z=0) - u_\phi(x_{N,k}, y_{N,k}, z=0) - \Delta V_{k} ||^2 $$

Donde $\Delta V_{k}$ es la diferencia de potencial medida empíricamente entre los electrodos $M$ y $N$.

### 2.2. Función de Pérdida Física ($\mathcal{L}_{PDE}$)

Fuerza a las redes a respetar la Ley de Conservación de la Carga de Maxwell (Ecuación de Poisson en estado estacionario). Este residual se evalúa sin necesidad de datos verdaderos ("unsupervised") sobre un conjunto de puntos de colocación $\mathbf{r}_i$ muestreados aleatoriamente en todo el dominio $\Omega$.

$$ \mathcal{L}_{PDE} = \frac{1}{N_{col}} \sum_{i=1}^{N_{col}} \left| \left| \nabla \cdot (\sigma_\theta(\mathbf{r}_i) \nabla u_\phi(\mathbf{r}_i)) - q(\mathbf{r}_i) \right| \right|^2 $$

Expandiendo el operador divergencia mediante la regla del producto (facilitando la diferenciación automática):
$$ \nabla \cdot (\sigma \nabla u) = \nabla \sigma \cdot \nabla u + \sigma \nabla^2 u $$

> [!IMPORTANT]
> **Regularización de la Fuente ($q$)**: Una carga puntual analítica se define con la delta de Dirac $\delta(\mathbf{r})$, la cual causa una singularidad infinita en las derivadas espaciales de la red. Para que la PINN pueda derivar fluidamente en torno a los electrodos, se aproxima la inyección y extracción de corriente $I$ mediante una distribución Gaussiana con radio $\epsilon$:
>
> $$ q(\mathbf{r}) = I \cdot \delta_{\epsilon}(\mathbf{r} - \mathbf{r}_A) - I \cdot \delta_{\epsilon}(\mathbf{r} - \mathbf{r}_B) $$
>
> Donde $\delta_{\epsilon}(\mathbf{r} - \mathbf{r}_0) = \frac{1}{(\sqrt{\pi} \epsilon)^3} \exp\left( - \frac{||\mathbf{r} - \mathbf{r}_0||^2}{\epsilon^2} \right)$.

### 2.3. Condiciones de Frontera ($\mathcal{L}_{BC}$)

Para asegurar que el sistema físico esté bien condicionado y tenga una solución única.

**Condición de Neumann (Interfaz Aire-Tierra)**:
La corriente no puede escapar hacia la atmósfera ($z>0$), implicando un flujo nulo normal a la superficie en todo punto que no sea estrictamente un electrodo inyector.
$$ \mathcal{L}_{Neumann} = \frac{1}{N_{surf}} \sum_{\mathbf{r} \in \partial \Omega_{surf} \setminus \epsilon_{elec}} \left| \left| \frac{\partial u_\phi}{\partial z}\bigg|_{z=0} \right| \right|^2 $$

**Condición de Dirichlet (Bordes de Dominio)**:
El potencial debe decaer asintóticamente a cero a medida que nos alejamos de las fuentes hacia los bordes infinitos del subsuelo (x, y lejano; z profundo).
$$ \mathcal{L}_{Dirichlet} = \frac{1}{N_{bnd}} \sum_{\mathbf{r} \in \partial \Omega_{inf}} || u_\phi(\mathbf{r}) ||^2 $$

$$ \mathcal{L}_{BC} = \mathcal{L}_{Neumann} + \mathcal{L}_{Dirichlet} $$

### 2.4. Regularización Espacial de la Conductividad ($\mathcal{L}_{reg}$)

Para prevenir que la red $\sigma_\theta$ genere distribuciones suaves e irreales (típicas de las redes densas), inyectamos prioris geológicos a través de la Total Variation (Norma $L_1$ del gradiente espacial).

$$ \mathcal{L}_{reg} = \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} || \nabla \sigma_\theta(\mathbf{r}_i) ||_{TV} $$
$$ \mathcal{L}_{reg} = \frac{1}{N_{\Omega}} \sum_{i=1}^{N_{\Omega}} \left( \left| \frac{\partial \sigma_\theta}{\partial x} \right| + \left| \frac{\partial \sigma_\theta}{\partial y} \right| + \left| \frac{\partial \sigma_\theta}{\partial z} \right| \right) $$

Esta penalización fuerza que las derivadas de $\sigma$ tiendan fuertemente a cero (fondo homogéneo) permitiendo saltos agudos (blocky inversion) consistentes con anomalías geológicas densas y esféricas.

### 2.5. Conservación de Carga Local (Término de Flujo, $\mathcal{L}_{flux}$)

Garantiza la conservación estricta de la corriente inyectada en las vecindades inmediatas de los electrodos fuente (A) y sumidero (B), integrando la ley de Ohm $\mathbf{J} = -\sigma \nabla u$ sobre superficies de control $\partial B_\epsilon$.

$$ \mathcal{L}_{flux} = \left( \frac{1}{N_A} \sum_{i=1}^{N_A} \sigma_\theta(\mathbf{r}_i) \nabla u_\phi(\mathbf{r}_i) \cdot \mathbf{n} - \frac{I}{|\partial B_\epsilon|} \right)^2 + \left( \frac{1}{N_B} \sum_{i=1}^{N_B} \sigma_\theta(\mathbf{r}_i) \nabla u_\phi(\mathbf{r}_i) \cdot \mathbf{n} + \frac{I}{|\partial B_\epsilon|} \right)^2 $$

Donde:
- $N_A$ y $N_B$ son puntos muestreados sobre las superficies de control $\partial B_\epsilon$ alrededor de los electrodos A y B.
- $\mathbf{n}$ es el vector normal a la superficie de control.
- $|\partial B_\epsilon|$ es el área de dicha superficie de control (ej. semiesfera si los electrodos están en superficie).
