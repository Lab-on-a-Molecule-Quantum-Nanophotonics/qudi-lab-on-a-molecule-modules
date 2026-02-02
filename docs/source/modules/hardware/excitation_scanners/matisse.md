# Matisse laser for scanning

The matisse laser is intended to have its own computer with Qudi running on it. It has locally [a minimal hardware](#qudi.hardware.laser.matisse.MatisseCommander) running to talk to the laser, and we connect to it on the main experiment computer where the [main scanner](#qudi.hardware.interfuse.remote_matisse_scanner.RemoteMatisseScanner) runs.

```{contents} Table of Contents
:depth: 3
``` 

## Matisse proxy

```{statemachine} qudi.hardware.laser.matisse.MatisseCommander
```

```{autodoc2-object} qudi.hardware.laser.matisse.MatisseCommander
render_plugin = "myst"
```

## Remote matisse scanner

For now this is intended to use the [matisse proxy](#qudi.hardware.laser.matisse.MatisseCommander) on a remote computer, but could be modified to use it on the same computer if needed.

This hardware uses a state machine that is depicted below. The link are clickable and lead to the specific method documentation for the state.

```{statemachine} qudi.hardware.interfuse.remote_matisse_scanner.RemoteMatisseScanner
```

```{autodoc2-object} qudi.hardware.interfuse.remote_matisse_scanner.RemoteMatisseScanner
render_plugin = "myst"
```

## Tips and tricks

It can happen that the NI card gets stuck because of a previous configuration. The error message looks like:

```
Specified route cannot be satisfied, because it requires resources that are currently in use by another route.

Source Device: Dev1
Source Terminal: Ctr0InternalOutput
Destination Device: Dev1
Destination Terminal: PFI0

Required Resources in Use by
Source Device: Dev1
Source Terminal: Ctr2InternalOutput
Destination Device: Dev1
Destination Terminal: PFI0

Status Code: -89137
```

You can fix this either by resetting the hardware in NI Max, or by resetting the hardware using the Python console. The Matisse hardware uses the finite sampling interface. This means that the command line will look like:
```python
ni_finite_sampling_input.reset_hardware()
```

```{note}
The first part (`ni_finite_sampling_input`) will vary depending on the configuration. Look into the `harware` tab or your configuration file!
```