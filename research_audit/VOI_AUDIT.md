# VOI Audit

VOI is implemented in two ways:

Counterfactual replay: for up to `max_counterfactuals` selected CIUs, remove one CIU, rerun the receiver, evaluate success, then assign +1 when removal changes success to failure, -1 when removal changes failure to success, else 0.

Self-reflection: if original task succeeds, selected CIUs self-reported by the receiver receive +1 and other selected CIUs receive 0. If original task fails, all selected CIUs receive 0. The code comment mentions a harsh negative penalty but the implementation uses neutral 0 on failures.

The main runner uses counterfactual VOI only when `config.evaluator.counterfactual` is true and only for CAPS. Current SWE-bench configs set counterfactual false, so they use self-reflection. `default.yaml` also sets counterfactual false; `caps_synthetic.yaml` sets counterfactual true.

VOI is used to update LinUCB only for CAPS via `policy.update(..., voi_scores=voi_scores)`.
