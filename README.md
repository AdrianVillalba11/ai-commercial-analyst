# Analista comercial de Brío

Aplicación de inteligencia artificial que permite a un analista de Almena Foods hacer preguntas en lenguaje natural sobre el desempeño de la marca **Brío** en el primer semestre de 2026 y recibir respuestas **fundamentadas, verificables y con las fuentes a la vista**.

Corre **completamente en el computador del analista**, con un modelo de lenguaje local (Ollama). No usa servicios de pago ni envía información fuera del equipo.

> Los datos y documentos de entrada son sintéticos, entregados para esta prueba técnica.

## Qué hace

- **Calcula con código, no con el modelo.** Las variaciones de ventas entre T1 (ene–mar) y T2 (abr–jun) salen de pandas, con la fórmula a la vista. El modelo de lenguaje nunca hace aritmética.
- **Combina cifras y documentos.** Lee el CSV de ventas, un estudio de consumidor (PDF), observaciones de puntos de venta (Word) y un reporte comercial (PDF), y cita el fragmento exacto que respalda cada afirmación.
- **Separa hechos, hipótesis e información insuficiente.** Las cifras las escribe el código; las hipótesis se marcan como *no demostradas*; y lo que no se puede afirmar se dice explícitamente.
- **Contrasta el reporte comercial con los datos,** región por región, y detecta que el reporte no coincide (por ejemplo, ubica la mayor caída en Centro cuando los datos la ubican en Sur).
- **No afirma causalidad sin evidencia.** Ante "¿las promociones causaron la caída?", explica qué no pueden mostrar los datos ni los estudios.
- **Resiste instrucciones ocultas en los documentos.** Uno de los archivos contiene una nota que ordena a una IA ignorar los datos; el sistema la aísla, la muestra como alerta y no la usa.
- **Muestra lo que descartó.** Si el modelo escribe algo que no pasa las verificaciones, se quita de la respuesta y se lista aparte con el motivo.

## Cómo funciona, en resumen

```
Pregunta → Planificador (modelo, con esquema cerrado) → Cálculo con pandas
                                                     → Búsqueda en documentos (ChromaDB + embeddings)
        → Redactor (modelo) → Validador (código) → Respuesta con fuentes y descartes
```

El modelo interpreta la pregunta y redacta; el código calcula y verifica. El detalle está en [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

## Requisitos

| Requisito | Detalle |
|---|---|
| Sistema | Windows 11 (el único entorno en el que se probó) |
| Python | 3.13 |
| Ollama | Para correr los modelos localmente: <https://ollama.com/download> |
| GPU | NVIDIA con 8 GB de memoria (se probó con una RTX 4060). En CPU no se probó y sería mucho más lento |
| Memoria RAM | 16 GB |
| Disco | Unos 7 GB libres para los modelos |

## Instalación

**1. Instalar Ollama** desde <https://ollama.com/download> y dejarlo en ejecución (queda en la bandeja del sistema).

**2. Descargar los dos modelos** (una sola vez; son unos 6 GB):

```powershell
ollama pull llama3
ollama pull bge-m3
```

`llama3` redacta las respuestas y `bge-m3` convierte los textos en vectores para buscar por significado.

**3. Crear el entorno virtual e instalar las dependencias**, desde la carpeta del proyecto:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**4. (Opcional) Configuración.** No hace falta para empezar. Si quieres cambiar modelos o el modo de búsqueda, copia la plantilla y edítala:

```powershell
Copy-Item .env.example .env
```

El archivo `.env` está en `.gitignore`. No hay claves ni credenciales en ninguna parte del proyecto.

## Ejecución

### Aplicación web

```powershell
python run.py
```

Se abre sola en <http://127.0.0.1:8000>. Al iniciar carga los datos y los documentos (unos 2 segundos) y calienta los modelos en segundo plano; el indicador de la esquina superior dice *Calentando modelos…* y pasa a *Listo* cuando terminan. Para detenerla, `Ctrl + C`.

Opciones: `--port 8080` cambia el puerto y `--no-browser` evita que se abra el navegador.

La página ofrece cuatro preguntas de ejemplo, el avance paso a paso mientras responde, las fuentes como botones que abren el texto original, y paneles de **Calidad de datos** y **Alertas de seguridad**.

### Línea de comandos

Sin abrir la página:

```powershell
python ask.py "¿Cuál fue la región con mayor caída porcentual de Brío?"
```

Sin argumentos entra en modo interactivo. `--mode hybrid|semantic|lexical` cambia la búsqueda en documentos y `--json` imprime el resultado completo.

### Tiempos esperados

Con los modelos ya cargados, una pregunta tarda **entre 6 y 15 segundos** (casi todo es el modelo escribiendo). La primera pregunta tras una pausa larga puede tardar unos segundos más, porque Ollama descarga los modelos de la memoria tras un rato de inactividad.

## Pruebas

```powershell
python -m pytest -q -m "not ollama"
```

Corre las pruebas que **no necesitan Ollama** (unos 15 segundos). Los modelos se simulan con dobles de prueba.

```powershell
python -m pytest -q
```

Corre todo, incluidas las pruebas que usan los modelos reales (algo más de un minuto). Si Ollama no está disponible, esas pruebas se saltan solas.

**Evidencia de ejecución** en [docs/evidencia/](docs/evidencia/): el resultado completo de la suite y las respuestas del sistema a las preguntas de referencia. Se regeneran con:

```powershell
python scripts/generate_evidence.py
```

## Modalidad de demostración

**Ejecución desde mi computador.** No hay una URL pública: la demostración se hace con `python run.py` en el equipo, según las instrucciones anteriores.

## Configuración

Todas las variables son opcionales. Se leen del archivo `.env` (plantilla en `.env.example`).

| Variable | Valor por defecto | Para qué sirve |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Dirección de Ollama. Se usa `127.0.0.1` y no `localhost` porque en Windows `localhost` añade unos 2 segundos por petición |
| `LLM_MODEL` | `llama3` | Modelo que planifica y redacta |
| `EMBEDDING_MODEL` | `bge-m3` | Modelo que convierte textos en vectores |
| `LLM_TIMEOUT_SECONDS` | `120` | Tiempo máximo de espera por respuesta de Ollama |
| `LLM_NUM_CTX` | `4096` | Ventana de contexto del modelo, en tokens. El prompt se recorta para caber |
| `LLM_KEEP_ALIVE` | `30m` | Cuánto tiempo mantiene Ollama los modelos en memoria |
| `RETRIEVAL_MODE` | `semantic` | Búsqueda en documentos: `semantic`, `hybrid` (significado + palabras clave) o `lexical` |
| `CHROMA_PATH` | `.cache/chroma` | Dónde se guarda el índice vectorial |

## Estructura del proyecto

```
.
├── README.md
├── requirements.txt          # dependencias con versiones fijas
├── .env.example              # plantilla de configuración (sin secretos)
├── run.py                    # inicia la aplicación web
├── ask.py                    # preguntas desde la terminal
├── data/raw/                 # los cinco archivos de entrada originales
├── src/analyst/
│   ├── config.py             # configuración
│   ├── service.py            # el flujo completo de una pregunta
│   ├── data/                 # carga y limpieza del CSV; reporte de calidad
│   ├── analytics/            # cálculos T1 vs T2 con pandas
│   ├── docs/                 # lectura y fragmentación de PDF, Word y Markdown
│   ├── security/             # detección de inyección y cuarentena
│   ├── rag/                  # búsqueda léxica, semántica (ChromaDB) y fusión
│   ├── llm/                  # cliente de Ollama, planificador, prompts y redactor
│   ├── guardrails/           # hechos por código, contraste con el reporte, validador
│   └── api/                  # API (FastAPI)
├── web/                      # la página: HTML, CSS y JavaScript sin frameworks
├── scripts/                  # evaluación de respuestas y generación de evidencia
├── tests/                    # pruebas automatizadas
└── docs/                     # arquitectura, evaluaciones y evidencia
```

## Datos de entrada

Los cinco archivos originales están en `data/raw/` y no se modifican. La base de ventas trae 4.000 filas con problemas de calidad (duplicados, texto mal escrito, valores no numéricos, un mes inexistente, negativos). Se limpia con reglas genéricas: **se corrige solo lo que tiene una única lectura posible y se excluye lo ambiguo, sin adivinar**. Resultado: 3.994 filas usadas, 6 excluidas y 3 corregidas, cada una registrada con su motivo en el panel **Calidad de datos**.

## Solución de problemas

| Síntoma | Qué hacer |
|---|---|
| El indicador dice **Ollama apagado** | Abre la aplicación de Ollama o ejecuta `ollama serve` |
| Dice **Falta el modelo …** | Ejecuta el `ollama pull` que indica al pasar el cursor sobre el indicador |
| `python run.py` no encuentra módulos | No está activado el entorno virtual: ejecuta el paso 3 de la instalación |
| El puerto 8000 está ocupado | `python run.py --port 8080` |
| La primera pregunta tarda mucho | Es el modelo cargándose; espera a que el indicador diga *Listo* o haz una pregunta de calentamiento |
| Las respuestas son lentas en toda la sesión | Cierra programas que usen la GPU (juegos, lanzadores): si falta memoria de vídeo, Ollama pasa capas a la CPU |
| Una pregunta falla con un mensaje de Ollama | El mensaje dice qué hacer; las causas habituales son Ollama apagado o un modelo sin descargar |

## Privacidad y seguridad

- Todo se ejecuta en el equipo. La aplicación solo escucha en `127.0.0.1`.
- La telemetría de ChromaDB está desactivada.
- No hay claves, credenciales ni servicios externos. La página no carga ningún recurso de internet.
- El texto que viene de documentos o del modelo se muestra siempre como texto plano, nunca como HTML.
- Las instrucciones ocultas dentro de los documentos se detectan, se aíslan y no llegan al modelo. Véase [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md).

## Documentación adicional

- [docs/ARQUITECTURA.md](docs/ARQUITECTURA.md): flujo, módulos, decisiones, seguridad y **limitaciones conocidas**.
- [docs/EVALUACIONES.md](docs/EVALUACIONES.md): cómo se eligieron los modelos y los métodos, con los resultados medidos.
- [docs/evidencia/](docs/evidencia/): ejecución completa de las pruebas y respuestas de referencia.
