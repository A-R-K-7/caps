# VOI Information-Boundary Audit

## Findings
The legacy `CAPS-VOI` implementation used `_derive_synthetic_voi` which directly accessed the `ground_truth_necessity` labels. This constituted a critical data leak.

## Remediation
1. Renamed the legacy oracle condition to `CAPS-OracleVOI`.
2. Rewrote `CAPS-VOI` to use the receiver's self-reported `used_ciu_ids` as the VOI signal, which relies strictly on visible information.
