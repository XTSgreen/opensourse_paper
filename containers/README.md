# Container scope

The Dockerfile provides the clean-room Python environment for the package, tests, B1 and figure generation. External method families with incompatible official dependencies are executed by the pinned method-level environments under `external_methods/environments/`; those environments are intentionally not copied into the release image.
