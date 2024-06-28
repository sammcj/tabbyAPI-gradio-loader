FROM debian:bookworm-slim

RUN apt-get update && \
  apt-get install -y build-essential python3 python3-pip git libcurl4-openssl-dev git curl

RUN git clone --depth=1 https://github.com/sammcj/tabbyAPI-gradio-loader /app

WORKDIR /app

RUN pip3 install --upgrade pip --break-system-packages

RUN pip3 install -r requirements.txt --break-system-packages

ADD entrypoint.sh /app/entrypoint.sh

ENV ENDPOINT_URL=${ENDPOINT_URL:-}
ENV AUTH=${AUTH:-}

EXPOSE 7860
ENTRYPOINT [ "/app/entrypoint.sh" ]
