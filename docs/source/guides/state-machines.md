# Guide: Writing state machines for hardwares or logics

For convenience, it can be useful to implement a hardware or a logic as a [finite state machine](https://en.wikipedia.org/wiki/Finite-state_machine) that runs in the background. This can be done by implementing the [`SampledFiniteStateInterface`](../modules/interface/sampled_finite_state_interface). You can refer to the linked documentation for details. An example of finite state interface implementation can be found in the `ExcitationScannerDummy` hardware documented below.

It implements the following finite state machine:
```{statemachine} qudi.hardware.dummy.excitation_scanner_dummy.ExcitationScannerDummy
```

```{autodoc2-object} qudi.hardware.dummy.excitation_scanner_dummy.ExcitationScannerDummy
render_plugin = "myst"
```

