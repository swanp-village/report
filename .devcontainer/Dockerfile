# syntax=docker/dockerfile:1.3-labs
ARG VARIANT="3.9"
FROM mcr.microsoft.com/vscode/devcontainers/python:0-${VARIANT}
ARG INSTALL_TEXLIVE="true"
RUN <<eot
if [ "${INSTALL_TEXLIVE}" = "true" ]
then
  apt-get update
  apt-get install -y \
    cm-super \
    texlive \
    texlive-latex-extra \
    texlive-luatex
  apt-get clean
  rm -rf /var/lib/apt/lists/*
fi
eot
