FROM ghcr.io/nish261/infintwetalkjfign/infinitetalk-runpod:4371d55 AS runtime

WORKDIR /

COPY runpod_app/comfy_handler.py /handler.py
COPY runpod_app/comfy_entrypoint.sh /entrypoint.sh
COPY runpod_app/comfy_workflows/I2V_single.json /I2V_single.json
COPY runpod_app/comfy_workflows/I2V_multi.json /I2V_multi.json
COPY runpod_app/comfy_workflows/V2V_single.json /V2V_single.json
COPY runpod_app/comfy_workflows/V2V_multi.json /V2V_multi.json
COPY runpod_app/comfy_examples /examples

RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
