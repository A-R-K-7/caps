# Seed Audit
- `RandomPolicy`: Varies completely based on seed.
- `CAPS-NoVOI` / `CAPS-VOI`: Deterministic at initialization regardless of seed. Exploration noise affects execution during the run, but the initial precision matrices and bounds are identical.
- **Conclusion**: Seed plumbing works correctly. Random uses it; CAPS is initialized neutrally.
