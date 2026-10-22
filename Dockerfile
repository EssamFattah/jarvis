FROM python:3.12-slim
 
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY app ./app
# Make the code readable by the non-root user, whatever permissions the files had on the host.
RUN chmod -R a+rX /srv/app
 
# Run as a non-root user.
RUN useradd --create-home --uid 1000 helper && mkdir -p /data && chown helper /data
USER helper
 
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
 