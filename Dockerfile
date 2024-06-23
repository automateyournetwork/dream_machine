# Stage 1: Base CUDA Image with Python
FROM nvidia/cuda:12.5.0-devel-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-pip \
    git \
    libsndfile1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Create a virtual environment, install requirements, copy core app files to the working directory
FROM base AS builder
WORKDIR /app
ENV VIRTUAL_ENV=/app/.venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

COPY requirements.txt .
# Install packaging module before the rest of the requirements
RUN pip install --no-cache-dir packaging
RUN pip install --no-cache-dir wheel
# Install torch before the rest of the requirements
RUN pip install --no-cache-dir torch
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-opencv \
    ffmpeg libsm6 libxext6

# Stage 3: Add the app user, copy the venv and app from the builder image, and launch the app.
FROM base AS app
ARG APP_USERNAME=appuser
ARG APP_UID=1000
ARG APP_GID=1000

WORKDIR /app

RUN groupadd --gid ${APP_GID} ${APP_USERNAME} && \
    useradd --uid ${APP_UID} --gid ${APP_GID} -m ${APP_USERNAME} && \
    chown ${APP_USERNAME}:${APP_USERNAME} /app

COPY --from=builder --chown=${APP_USERNAME}:${APP_USERNAME} /app ./
COPY --chown=${APP_USERNAME}:${APP_USERNAME} dream_machine.py ./
COPY --chown=${APP_USERNAME}:${APP_USERNAME} image_selections.yaml ./
COPY --chown=${APP_USERNAME}:${APP_USERNAME} audio_selections.yaml ./
USER ${APP_USERNAME}
ENV VIRTUAL_ENV=/app/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

# The following COPY commands are placed at the end to ensure they are always run
COPY --chown=${APP_USERNAME}:${APP_USERNAME} dream_machine.py ./
COPY --chown=${APP_USERNAME}:${APP_USERNAME} image_selections.yaml ./
COPY --chown=${APP_USERNAME}:${APP_USERNAME} audio_selections.yaml ./

CMD ["streamlit", "run", "dream_machine.py", "--server.port", "8501"]
