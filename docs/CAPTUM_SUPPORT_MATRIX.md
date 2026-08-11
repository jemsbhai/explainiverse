# Captum-backed explainer support matrix

This matrix is a fail-closed compatibility contract, not a claim that Captum
supports arbitrary PyTorch graphs. Explainiverse declares `captum>=0.8,<1.0`;
the reviewed lower endpoint is 0.8.0 and the current lock resolves 0.9.0. Both
versions must pass the same tests before a release. A passing current-only run
does not establish lower-bound support.

Captum itself documents that hook-based DeepLIFT/DeepLiftShap requires explicit
activation modules and does not safely support reuse of one nonlinear module.
See the [Captum FAQ](https://captum.ai/docs/faq) and
[DeepLIFT API](https://captum.ai/api/deep_lift.html). Captum 0.9.0 describes
upgrading from 0.8 as drop-in for most users, but that release statement is not
a substitute for this repository's parity gates; see the
[Captum releases](https://github.com/meta-pytorch/captum/releases).

## Version and API dependency

| Surface | Captum 0.8.0 | Captum 0.9.0/current | Explainiverse contract |
|---|---|---|---|
| `captum.attr.DeepLift` | Public | Public | Required backend for `DeepLIFTExplainer`; no ordinary-gradient fallback |
| `captum.attr.DeepLiftShap` | Public | Public | Required backend for background-distribution `DeepLIFTShapExplainer` |
| `captum.attr.LRP` | Public | Public | Required backend for epsilon, gamma, z-plus, and composite rules |
| `PropagationRule`, `EpsilonRule`, `GammaRule`, `Alpha1_Beta0_Rule`, `IdentityRule` | Private `captum.attr._utils.lrp_rules` | Same private module | Required to attach method-specific rules; import availability and parity are tested at both endpoints |

The LRP rule classes are not exported by `captum.attr.__all__` in either
0.8.0 or 0.9.0. Captum's public `LRP` accepts rule objects through module
attributes but exposes no public propagation-rule base or constructors.
Consequently `lrp.py` still has one isolated private import. Removing it by
substituting unrelated public classes would change the algorithm. If a future
Captum version adds a public rule-extension API, migrate only after the direct
rule parity gates pass. If the private symbols disappear first, LRP must fail
at import/construction; it must not fall back to another rule.

## DeepLIFT and DeepSHAP graph matrix

The public input is one flat feature vector. The caller model must be an exact
`nn.Sequential` graph (nested exact Sequentials are allowed) or one exact
supported leaf. Arbitrary custom `forward` programs are not accepted: FX
tracing cannot prove tensor execution when Python type branches, leaf spoofing,
or runtime mutation are present. Every accepted module type is exact, not an
`isinstance` subclass match.

| Graph component | DeepLIFT Rescale | DeepSHAP background API | Boundary |
|---|---:|---:|---|
| `Linear`; 1D/2D/3D convolution or transposed convolution | supported | supported | Only when reachable from the flat input through the declared graph |
| BatchNorm 1D/2D/3D; identity; flatten/unflatten | supported | supported | BatchNorm must track non-`None` running mean/variance |
| Dropout family; average/adaptive-average pool; constant pad | supported | supported | Evaluated under restored model state |
| ReLU, ELU, LeakyReLU, Sigmoid, Tanh, Softplus; MaxPool 1D/2D/3D without returned indices | supported Rescale modules | supported Rescale modules | Each nonlinear module object may execute once only |
| `nn.Softmax` in the caller graph | rejected before attribution | rejected before attribution | Captum's coupled-output result is not conservative on the verified completeness oracle |
| Multiclass adapter softmax with `gradient_output="prediction"` | rejected | rejected | Explain raw/model scores instead; the single-logit complementary-sigmoid contract is separate |
| Functional activations, custom forwards, GELU, subclasses, or another unlisted module | rejected before attribution | rejected before attribution | No FX or gradient-times-input approximation |
| Dynamic/untraceable control flow | rejected before attribution | rejected before attribution | Arbitrary custom roots are outside the supported graph |
| Reused nonlinear module | rejected before attribution | rejected before attribution | Separate module instance per use site |
| Single-baseline setter/helper inherited by DeepSHAP | not applicable | `NotImplementedError` | Use `set_background` / background constructor |
| Ordinary-IG comparison inherited by DeepSHAP | available on DeepLIFT only | `NotImplementedError` | No background-expected comparator is defined |

At construction and again before every model/Captum forward, Explainiverse
checks the exact registered topology and module identities, immutable
module-import-time canonical `forward`, `_call_impl`, and
`_wrapped_call_impl` identities, canonical Captum traversal/hook-registration
methods, and empty local forward-pre/forward/backward-pre/backward hook
registries. The exact root `Sequential` is checked too. Any process-global
PyTorch module execution hook also fails closed at that boundary. A compiled
call implementation is outside this contract. Post-construction child
replacement, method shadowing/monkeypatching, or hook registration requires a
new explainer; it is never silently accepted.

## LRP graph and rule matrix

All LRP graphs are one exact `nn.Sequential` leaf chain (or one supported leaf).
Nested arbitrary containers, branches, residual additions, functional
operations, and shared modules are rejected during construction.

| Layer family | epsilon / gamma / z-plus / composite | native alpha-beta |
|---|---:|---:|
| `Linear` | supported | supported |
| `Conv2d` | supported | rejected |
| BatchNorm 1D/2D with tracked running statistics | supported | rejected |
| MaxPool2d without returned indices; AvgPool2d; AdaptiveAvgPool2d | supported | rejected |
| ReLU, Tanh, Dropout, LeakyReLU, ELU, Sigmoid | supported | supported |
| Flatten, Unflatten | supported | supported |
| Reused layer object, any unlisted leaf, untracked BatchNorm, MaxPool indices | rejected | rejected |

LRP applies the same exact canonical call-pipeline and no-execution-hook checks
to the root and every leaf at construction and every compute. Because Captum
LRP snapshots through `state_dict()`/`load_state_dict()`, LRP additionally
requires canonical state-I/O methods and empty state/load-state hook
registries. Its cached leaf order and identities must still match the live
registered graph before propagation begins.

One-output classification remains separately restricted. Explainiverse never
derives class 0 by negating a sign-asymmetric gamma, z-plus, alpha-beta, or
composite result. Probability-output graphs are not relabelled as raw-logit
LRP.

## Mandatory compatibility gates

Run the following twice: once after installing exact `captum==0.8.0`, and once
with the lock/current version (currently 0.9.0). Each environment must import
Captum explicitly before pytest and must report zero skips in these files.

```bash
python -c "import captum; print(captum.__version__)"
pytest --strict-config --strict-markers \
  tests/test_deeplift.py tests/test_lrp.py tests/test_lrp_accuracy.py \
  tests/reference/test_ref_deeplift.py \
  tests/test_gradient_approved_remediation.py
```

The release gate must cover, at minimum:

- analytical linear/piecewise DeepLIFT completeness and direct Captum parity;
- DeepSHAP background averaging and conservation where defined;
- functional, untraceable, and unsupported graph rejection before an
  explanation is returned; reused nonlinear DeepLIFT modules and every shared
  LRP module reject, while shared linear DeepLIFT modules remain supported;
- epsilon, gamma, z-plus, composite, and native alpha-beta analytical/reference
  parity for every declared layer family;
- success/failure restoration of hooks, training flags, buffers, gradients,
  default/custom RNG state, and explainer configuration;
- all quarantined DeepSHAP helpers continuing to raise `NotImplementedError`.

CUDA parity remains outside this matrix until the dedicated hardware gate is
green. This file does not broaden any registry claim beyond CPU verification.
