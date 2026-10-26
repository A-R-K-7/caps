# Solver/Receiver Audit
The `ReceiverAgent` correctly takes only the `selected_cius`.
- When tested with a Smart Mock (returns the expected answer), the receiver outputs a JSON that passes the evaluator.
- No ground truth labels are passed to the receiver prompt. It only sees CIU text and CIU IDs.
