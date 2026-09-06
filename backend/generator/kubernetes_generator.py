def kubernetes_manifest_content(package_name="generated", env_vars=None):
    """The exact kubernetes.yaml text generate_kubernetes_manifest (below)
    writes to disk, as a pure string -- no filesystem access at all. See
    dockerfile_content's own docstring (backend/generator/docker_generator.py)
    for why this split exists.

    A compiled app already gets a Dockerfile, a docker-compose.yml for a
    single-host `docker compose up`, a .env.example, and a README -- but
    nothing for the far more common real deployment target those docstrings
    already name in passing (GET /api/env-vars-preview's own docstring: "An
    operator writing a docker-compose.yml, a Kubernetes manifest, or a plain
    .env file... had no way to discover any of this except by trial and
    error"): a Kubernetes cluster. An operator deploying there had to
    hand-write a Deployment/Service manifest from scratch, transcribing the
    Dockerfile's own $PORT/HEALTHCHECK contract and every NOTEBOOK_API_*
    variable GET /api/env-vars-preview already reports one field at a time,
    with nothing to catch a typo'd name, a stale default, or a probe path
    that's drifted from the app's own actual health/readiness routes.

    `env_vars` is GENERATED_APP_ENV_VARS (backend/generator/
    api_generator.py) -- the exact same list docker_compose_content/
    env_example_content already take (see docker_compose_content's own
    docstring for why it's passed in rather than imported directly here, to
    avoid a circular import) -- so the container's own "env:" list can
    never list a variable the compiled app doesn't actually recognize, or a
    default that's drifted from what it would actually fall back to.

    Each entry becomes a literal "name: value" pair rather than a
    shell-style "${NAME:-default}" the way docker_compose_content's own
    "environment:" section does -- Kubernetes env values aren't expanded by
    a shell at all, so there's no equivalent syntax; an operator overrides
    one by editing this manifest directly (or layering a Kustomize/Helm
    values file over it), the same "a real, valid file on its own, only a
    value you actually want to override needs editing" precedent
    env_example_content's own docstring already sets. NOTEBOOK_API_KEY in
    particular ships here as a plain literal for the same reason
    env_example_content leaves it as one: every value already matches the
    compiled app's own real default -- moving it into a Secret is a real
    deployment's job, not this preview's.

    "PORT" -- read by the Dockerfile's own CMD/HEALTHCHECK, never by the
    compiled app itself (see GET /api/env-vars-preview's own docstring for
    why it's deliberately excluded from GENERATED_APP_ENV_VARS) -- gets the
    identical unconditional inclusion docker_compose_content's own
    "environment:" section and env_example_content's own file already give
    it, driving the container's own containerPort/probe ports so they can
    never drift from what the container itself actually binds to.

    livenessProbe/readinessProbe target the compiled app's own GET
    /health/GET /ready -- neither requires an X-API-Key header (see
    RESERVED_INFRASTRUCTURE_NAMES in api_generator.py: health_check/
    readiness_check are the only two built-in routes with no
    Depends(verify_api_key)), the same unauthenticated endpoints the
    Dockerfile's own HEALTHCHECK already curls, so a probe here needs no
    credential this manifest would otherwise have to embed.

    Renders as two YAML documents separated by "---" (a Deployment, then a
    ClusterIP Service) -- the minimal pair `kubectl apply -f` needs to
    actually run and reach the compiled app inside the cluster; anything
    beyond that (an Ingress, a HorizontalPodAutoscaler, resource
    requests/limits) is a real deployment's own decision this tool has no
    way to make on an operator's behalf.
    """
    env_vars = env_vars or []

    env_entries = [{"name": "PORT", "default": "8000"}] + list(env_vars)

    env_lines = "\n".join(
        f'            - name: {entry["name"]}\n'
        f'              value: "{entry["default"]}"'
        for entry in env_entries
    )

    return f"""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {package_name}
  labels:
    app: {package_name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {package_name}
  template:
    metadata:
      labels:
        app: {package_name}
    spec:
      containers:
        - name: {package_name}
          image: {package_name}:latest
          ports:
            - containerPort: 8000
          env:
{env_lines}
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 30
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: {package_name}
spec:
  selector:
    app: {package_name}
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
"""


def generate_kubernetes_manifest(
    output_path="generated/kubernetes.yaml",
    package_name="generated",
    env_vars=None,
):
    """Write a kubernetes.yaml for the compiled app at `output_path`,
    alongside the Dockerfile/.dockerignore/docker-compose.yml/.env.example/
    README.md generate_dockerfile/generate_dockerignore/
    generate_docker_compose/generate_env_example/generate_readme already
    write there on every compile -- see kubernetes_manifest_content's own
    docstring above for why this exists and what it contains.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(kubernetes_manifest_content(package_name, env_vars))

    print(f"kubernetes.yaml generated at: {output_path}")
