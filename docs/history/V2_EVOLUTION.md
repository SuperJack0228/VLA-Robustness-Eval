# MiniVLA V2 Engineering Evolution

## 1. Purpose of This Record

This document records the complete engineering path from the first MiniVLA V2
system to the frozen V2 Clean standard. It deliberately includes unsuccessful
experiments. The project improved most when an attractive offline metric was
treated as a hypothesis rather than proof of closed-loop competence.

The final standard described here is:

- Dataset version: `v2.clean`
- Schema version: `5`
- Model architecture version: `4`
- Checkpoint format: `7`
- Demonstrations: 1200, balanced as 200 per task
- Final policy: `artifacts/v2-clean-rc1/mini_vla_v2_clean_policy.pth`
- Selected checkpoint epoch: 18
- Clean evaluation: 224 / 240 across two independent seeds, or 93.33%
- 95% Wilson confidence interval: approximately 89.45% to 95.86%

Generated datasets and checkpoints are local artifacts under `results/` and
are intentionally excluded from Git.

## 2. Version Overview

| Version | Main goal | Principal result | Decision |
|---|---|---|---|
| V2 | Establish the six-task language-conditioned architecture | 12.5% clean success | Full-chain diagnosis required |
| V2.1 | Repair initial-state grounding and training/evaluation mismatch | 72.5% clean, 75.83% task success | Architecture validated, manipulation still weak |
| V2.2 | Repair contact physics, expert behavior, and closed-loop stability | Much smarter motion; approximately 70% observed baseline | Use as stable rollback and perception warm-start |
| Stage 1 / 1.1 execution experiments | Test whether runtime latching and alignment could rescue failures | Visually improved results, especially Pick | Rejected as final scoring because it used privileged state |
| V2.3 | Add aggressive recovery demonstrations and stronger contact behavior | Severe Pick regression after retraining | Dataset and checkpoints deleted |
| Conservative rollback attempt | Return to V2.2 and apply narrow corrections | Grasped objects but frequently failed to lift | Root transition-label issue still unresolved |
| V2 Clean | Rebuild causal transitions, interaction conditioning, raw evaluation, and gates | 93.33% over 240 clean episodes | Frozen final standard |

## 3. Foundation Before V2

Before the V2 line, the project established the physical and software stack:

1. Official MuJoCo Python bindings were validated on macOS without
   `mujoco-py`.
2. Panda `Lift` ran in robosuite with native `action_spec` sampling.
3. A four-phase scripted Pick expert was implemented: approach, descend,
   close, and lift.
4. The first collector saved RGB observations, EEF state, and 4D actions.
5. A small ACT-style model demonstrated basic spatial approach behavior.

That MVP proved the full I/O path, but it was not yet a credible multi-task
VLA baseline. It used a restricted action space, short demonstrations, weak
language conditioning, and insufficient failure diagnostics.

## 4. V2: Six-Task Multimodal Foundation

### 4.1 Objective

V2 changed the problem from single-object Lift imitation into a balanced
language-conditioned manipulation benchmark with six tasks:

- Pick red cube, blue ball, or green cylinder.
- Push red cube, blue ball, or green cylinder away from the robot.

### 4.2 Main architectural changes

V2 introduced the foundations retained by every later version:

- Three objects simultaneously present in every scene.
- 7D `OSC_POSE` delta control: XYZ, RPY, and gripper.
- `agentview` and `robot0_eye_in_hand` cameras at 112 x 112.
- A 17D proprioceptive state containing EEF position, quaternion, gripper
  position and velocity, and finite-difference EEF linear/angular velocity.
- Five-step state history.
- Twenty-step ACT action chunks.
- Frozen DistilBERT instruction encoding.
- Shared ResNet50 visual encoding for both views.
- GroupNorm replacement for all BatchNorm2d layers.
- Separate continuous pose and binary gripper outputs.
- Phase, target-position grounding, and target-class supervision.
- Episode-interleaved batches so one batch did not consist of consecutive
  frames from one trajectory.

### 4.3 Staged engineering process

The first V2 implementation was organized into five gates:

- Stage 0: deterministic oracle and scene integrity.
- Stage 1: balanced demonstration collection.
- Stage 2: multimodal model and training pipeline.
- Stage 3: closed-loop raw policy evaluation.
- Stage 4: preflight, postflight, and final benchmark checks.

The first balanced dataset contained 600 episodes. This was enough to reveal
systemic faults quickly without spending the time required for 4800 episodes.

### 4.4 First closed-loop result

The first formal V2 evaluation achieved only 12.5% success, with 15 successes
in 120 episodes. Dominant failures were:

- `target_not_reached`: 35
- `target_not_contacted`: 48
- `missed_grasp`: 19
- Additional drops and insufficient lifts

The low score was not explained by language classification alone. The model
could often identify the requested target but failed to produce a correct
initial trajectory and contact sequence.

### 4.5 Root causes discovered

The first diagnosis identified several interacting problems:

1. Initial frames were a tiny fraction of all training windows, while every
   evaluation episode starts from an initial frame.
2. Validation averaged thousands of easy mid-trajectory windows and hid poor
   first-action behavior.
3. Highly correlated samples still leaked into batches despite a nominal
   shuffled loader.
4. Target grounding could be correct while the action decoder ignored or
   weakly used the grounding representation.
5. Small-batch visual training made BatchNorm statistics unstable.
6. A low aggregate action loss did not guarantee the correct phase sequence.

The key lesson from V2 was that ordinary validation loss was not a sufficient
policy-release criterion.

## 5. V2.1: Initial-State and Grounding Rescue

### 5.1 Changes

V2.1 addressed the train/evaluation distribution mismatch:

- Initial windows were explicitly oversampled.
- A dedicated initial-state validation loader was added.
- Initial XYZ error, grounding error, target selection, and target class were
  included in model selection.
- Episode-aware interleaving guaranteed many independent episodes and tasks in
  every batch.
- Visual augmentation was restricted to physically valid changes: mild color
  jitter, blur, and translation without flips or rotations.
- ResNet50 started from ImageNet weights.
- Early ResNet stages were frozen and BatchNorm was replaced by GroupNorm.
- Learning rate was reduced and weight decay increased.
- Per-task first-action metrics were written to the persistent CSV log.

### 5.2 Result

The V2.1 clean evaluation improved dramatically:

- Task success: 75.83%
- Clean success: 72.5%
- Target class accuracy: 100%
- Wrong-object contact: 3.33%

The remaining failure counts included:

- Missed grasp: 14
- Insufficient push distance: 11
- Insufficient lift: 4
- Wrong-object contact: 4

This was the first strong evidence that the multimodal architecture itself was
viable. The remaining errors were concentrated around physical contact and
temporal execution rather than language identity.

### 5.3 Problems exposed by visualization

Visual runs showed issues that aggregate CSV metrics could not fully explain:

- The blue ball could penetrate the table under unfavorable contact settings.
- Push-B approached correctly but contacted too high or pushed upward.
- The arm sometimes reached workspace limits and oscillated.
- A toppled cylinder invalidated the original push geometry.
- Red-cube grasps sometimes contacted a top corner instead of centering the
  fingers.
- Pick trajectories could grasp, lift briefly, release, and retry.

These observations motivated a new expert dataset instead of another optimizer
tuning pass.

## 6. V2.2: Contact Physics and Expert Stabilization

### 6.1 Scene and physics changes

V2.2 focused on making expert behavior physically consistent:

- The blue ball radius was increased to 2.6 cm.
- Sphere friction, damping, `solref`, and `solimp` were tuned to prevent table
  penetration and unstable impulses.
- Push contact heights became object specific.
- Push actions were made horizontal or slightly downward during contact.
- Cylinder uprightness and table retention became explicit diagnostics.
- Collision-safe push corridors were checked before accepting a scene.

### 6.2 Oracle changes

- Grasp confirmation and lost-grasp confirmation were made less sensitive to
  one-frame contact noise.
- Pick and Push recovery states were added.
- Push goals used deterministic directions derived from robot and object
  geometry.
- Push-B contact heights were benchmarked across multiple candidates.
- Retry trajectories attempted to reacquire moved targets.
- Accepted data remained balanced across all six tasks.

### 6.3 Collection and schema incident

During V2.2 collection, a failed trajectory reached archive construction with a
nonzero `success_step`. The schema correctly rejected it with:

`Failed trajectories must use success_step=0`

The collector was fixed so diagnostic failures and accepted successful
episodes used separate, internally consistent metadata. Collection then
completed with 1200 balanced episodes.

### 6.4 Training and observed behavior

The V2.2 model stopped early after validation stopped improving. Visual
behavior was markedly better:

- No violent arm twitching.
- Reliable target selection.
- Much fewer empty pushes and random grasps.
- Strong Pick-B and Pick-C performance.

However, the measured baseline remained around the high-60% to low-70% range
in the project evaluations. Persistent weaknesses were:

- Pick objects could be released after a successful initial lift.
- The policy could return to a stale grasp location after the target moved.
- Push-B often stopped before the displacement threshold.
- Push-C did not always respond coherently after the cylinder toppled.

V2.2 therefore became the stable rollback point, not the final standard.

## 7. Stage 1 and Stage 1.1 Runtime Assistance Experiments

### 7.1 Why they were attempted

The team tested whether conservative execution logic could separate model
quality from known control failure modes. Experimental runtime features
included:

- Gripper latching after true bilateral grasp.
- Clearing stale action chunks when contact or grasp state changed.
- Replanning after observed target motion.
- Push-B contact-height guards.
- Push-B lateral realignment using predicted and true geometry.
- Phase overrides that continued Phase 7 while push distance was insufficient.

### 7.2 Outcome

Pick performance improved substantially even though the checkpoint did not
change. That was useful diagnostically: the network often reached a graspable
state, and execution timing caused many failures.

It was not acceptable as the final baseline because some decisions used
simulator-only contact, grasp, or object state. Such assistance changes the
question from "Can the VLA policy solve the task?" to "Can the policy plus a
privileged state machine solve the task?"

The Stage 1.1 Push-B guard also failed to solve the underlying push-distance
problem. Runtime correction could improve a particular contact but could not
teach long-horizon task progress.

### 7.3 Decision

All privileged execution modes were removed from the final evaluator. The
current CLI accepts only `--execution-mode raw`. The only remaining shield uses
the robot's own EEF workspace and table floor; it does not steer toward true
object coordinates or force the gripper based on simulator grasp state.

## 8. V2.3: Aggressive Recovery Dataset and Regression

### 8.1 Intended repairs

V2.3 attempted to solve every remaining issue in one dataset revision:

- Five-step grasp confirmation.
- Five-step lost-grasp confirmation.
- Forced Pick recovery and displaced re-localization trajectories.
- Blue-ball contact-height scanning.
- Strictly horizontal Phase 7 pushes.
- Continue pushing toppled cylinders or label them as failures.
- More recovery examples and stronger contact supervision.

### 8.2 Collection-time failures

The stricter schema exposed two concrete defects:

1. Push-B episodes were rejected because contact was systematically too high:
   `p90=0.0341m`.
2. Preflight failed because Pick-C lacked enough 2-5 cm displaced
   re-localization trajectories.

Both incidents showed the value of schema-level semantic tests. They prevented
known bad archives from silently entering training.

### 8.3 Training regression

After collection and training completed, visual Pick-A immediately showed a
severe regression: the gripper captured the red cube but then stopped instead
of lifting. Continuing all six visual tasks was unnecessary because the core
Pick transition was visibly broken.

### 8.4 Forensic audit

The audit found that the aggressive recovery implementation had made the
labels internally contradictory:

- A physically grasped object could remain labeled Phase 2 for several steps.
- Those confirmation steps still contained negative Z actions.
- The model therefore saw "grasped" followed by "press downward or hold" before
  seeing Phase 3 lift.
- Forced recovery trajectories increased the frequency and influence of this
  ambiguous boundary.
- Independent pose, phase, and gripper heads could each score well while their
  joint transition was wrong.
- Initial-state and aggregate validation gates did not inspect the exact first
  action after grasp acquisition.

The problem was not insufficient model capacity. It was a causal label error at
the most important contact transition.

### 8.5 Decision

The V2.3 dataset and checkpoints were deleted. Mixing V2.3 with older datasets
was explicitly rejected because it would preserve contradictory labels. The
project returned to the V2.2 code line and reused only trusted perception
weights.

## 9. Conservative Rollback Attempt

The first rollback deliberately avoided another large redesign. V2.2 behavior
was restored and only narrow fixes were applied. Training completed, but the
new evaluation still showed red-cube grasps that did not transition into lift.

This result was important. It proved that merely deleting the most complex
V2.3 recovery branches did not remove the underlying transition ambiguity from
the full pipeline. A clean rebuild of expert labels, sampling, model coupling,
and release gates was required.

## 10. V2 Clean: Causal Release Redesign

### 10.1 Design rule

V2 Clean adopted one strict principle:

> Every accepted action label must describe what the robot should do next from
> the physical state recorded in that same sample.

The implementation avoided broad runtime assistance and concentrated on clean
expert causality.

### 10.2 Clean Pick state machine

Pick uses phases 0 through 4:

- Phase 0: move above the live target.
- Phase 1: descend to the grasp pose.
- Phase 2: close the gripper and confirm bilateral grasp.
- Phase 3: lift while holding the gripper closed.
- Phase 4: release, retreat vertically, reacquire the live target, and retry.

Critical fixes:

- Once `env.is_grasping(target)` becomes true in Phase 2, the position target is
  set to the current EEF position. The expert no longer presses a grasped object
  into the table.
- Grasp confirmation is two steps, not five.
- The next recorded phase becomes Phase 3 within two steps.
- Phase 3 uses a dedicated positive lift gain.
- A real grasp always overrides a scripted recovery and immediately restores
  Phase 3.
- Lost-grasp detection still uses five steps to reject contact jitter.

### 10.3 Clean Push state machine

Push uses phases 5 through 9:

- Phase 5: approach behind the live object.
- Phase 6: descend to object-specific contact height.
- Phase 7: push along the scheduled direction.
- Phase 8: hold after verified task success.
- Phase 9: retreat vertically and recompute the approach from the live object
  position before retrying.

Critical fixes:

- Push retries no longer reuse a stale target point.
- Push-B has a 30% forced lateral-offset recovery curriculum.
- The forced miss lasts eight steps and then teaches retreat and visual
  reacquisition.
- Phase 7 remains horizontal at contact.
- Phase 8 is entered only after the environment's real displacement condition
  is satisfied in the expert.
- Push success must remain stable for five steps before archive acceptance.

### 10.4 Collision integrity

Physical placement validity alone was not enough. The Panda fingers can sweep a
distractor even when object geoms do not initially overlap. V2 Clean adds:

- 5.0 cm extra target clearance for bilateral grasps.
- 6.0 cm push-corridor margin for cube and ball.
- 8.0 cm push-corridor margin for the cylinder because a falling cylinder has
  a less predictable swept envelope.

Only clean successes are saved. Failed attempts, wrong-object contacts,
physics failures, and schema failures are discarded.

### 10.5 Schema version 5

The schema validates shape, dtype, metadata, and causal manipulation signals.
For successful Pick trajectories it additionally requires:

- A real grasp event.
- A later lift event.
- Grasp-to-lift delay of at most two steps.
- No strong negative Z action while grasped in Phase 2.
- Positive first Phase 3 Z action of at least 0.20.
- Closed gripper at the lift transition.

For Push-B it checks contact height and forward action statistics. The schema
also verifies task balance, split integrity, target identity, success-step
semantics, and visual tensor dtypes.

### 10.6 Final dataset

Collection seed: `20260817`.

- Total accepted episodes: 1200
- Train: 960
- Validation: 120
- Test: 120
- Each split remains task balanced.
- Each of the six full-dataset buckets contains 200 accepted episodes.
- Image tensors remain `uint8` at 112 x 112.
- Actions are 7D OSC_POSE deltas.
- State is 17D and action chunks are length 20.

The final Preflight passed all four gates:

- `archive_schema_balance_and_signal`
- `train_only_normalization`
- `interleaved_dataloader`
- `model_optimizer_checkpoint`

### 10.7 Transition-aware data loader

The final loader does not treat every frame as equally informative:

- Successful suffixes exclude failed early attempts while retaining the final
  recovery and successful attempt.
- Phase-balanced sampling increases contact and manipulation coverage.
- Explicit windows are added around every phase transition.
- Pick lift transitions and Push recovery windows receive dedicated flags.
- Initial samples remain oversampled.
- Every batch mixes 16 episodes and multiple task buckets.
- Mild state noise and history dropout improve recovery robustness.
- Color jitter, blur, and small translations are allowed; flips and rotations
  remain forbidden because they change physical semantics.

The final training set contained 61,440 windows from 960 episodes. Validation
contained 5,760 windows from 120 episodes.

### 10.8 Model architecture version 4

The final model contains:

- Frozen DistilBERT language encoder and a trainable language projection.
- Shared ImageNet ResNet50 dual-view backbone.
- GroupNorm in place of every BatchNorm2d.
- Frozen stem, layer1, and layer2.
- Spatial visual tokens rather than a single pooled visual vector.
- Five-step normalized proprioceptive history.
- Target-position grounding decoder and target-class head.
- Two-logit interaction head for current target contact and current grasp.
- Interaction probabilities projected back into the action memory as a token.
- Phase prediction and a soft phase embedding added to each action query.
- Twenty learned action queries.
- Continuous 6D pose head and binary gripper-logit head.

The interaction and phase conditioning couple previously independent heads.
The model can no longer achieve a low pose loss while completely ignoring its
own predicted manipulation state.

### 10.9 Final training objective

The weighted objective combines:

- Pose L1 loss.
- Gripper BCE with transition and closed-hold weighting.
- Action-delta smoothness.
- Phase cross entropy.
- Contact/grasp BCE with positive-class weighting.
- Target grounding Smooth L1.
- Target class cross entropy.
- A dedicated Pick lift-transition margin requiring positive Z, closed
  gripper, Phase 3, and predicted grasp jointly.

Model selection combines full validation, initial-state validation, and lift
transition quality. Checkpoint format 7 stores model configuration,
normalization, schema version, dataset version, optimizer, scheduler, and all
release metrics.

Only 184 V2.2 perception and grounding tensors were warm-started. The policy
decoder, interaction modules, phase conditioning, and action heads were fresh.

### 10.10 Training result

Training stopped early at epoch 28 after patience reached 10. The exported
policy is the best checkpoint from epoch 18, not the final epoch.

Final logged epoch 28 metrics included:

- Train total: 0.17526
- Validation total: 0.19926
- XYZ MAE: 0.0167
- Gripper accuracy: 99.83%
- Phase accuracy: 97.77%
- Contact accuracy: 99.29%
- Grasp accuracy: 99.41%
- Lift Gate: 98.33%
- Initial grounding: 0.72 cm
- Initial target selection/class: 100% / 100%

### 10.11 Full-trajectory Postflight

The selected epoch-18 policy passed validation and test gates:

| Metric | Validation | Test |
|---|---:|---:|
| XYZ MAE | 0.01890 | 0.01842 |
| RPY MAE | 0.00793 | 0.00756 |
| Grounding | 0.80 cm | 0.83 cm |
| Target selection | 100% | 100% |
| Target class | 100% | 100% |
| Contact accuracy | 98.68% | 98.44% |
| Grasp accuracy | 99.25% | 99.46% |
| Lift-transition joint accuracy | 100% | 98.33% |

Every Pick bucket met the per-task lift-transition requirement. Pick-A test
Lift Gate was the lowest at 95%, which correctly predicted its slightly higher
closed-loop residual failure rate.

### 10.12 Raw evaluator cleanup

The final evaluator removed all Stage 1 and Stage 1.1 assistance:

- No simulator grasp-based gripper latch.
- No true object-position realignment.
- No contact-triggered forced replanning.
- No phase override based on measured push distance.
- No forced close after a previously observed grasp.

The output explicitly reports:

- `execution_mode: raw`
- `uses_privileged_execution_assistance: false`
- `score_scope: policy_with_workspace_safety_only`

The remaining shield only prevents commands from leaving the demonstrated EEF
workspace or descending below the object-specific table floor.

## 11. Final Closed-Loop Evidence

### 11.1 Visual six-task run

Ten rendered episodes per task produced 56 / 60 clean successes, or 93.33%:

- Pick-A: 10 / 10
- Pick-B: 10 / 10
- Pick-C: 9 / 10
- Push-A: 9 / 10
- Push-B: 10 / 10
- Push-C: 8 / 10

This run showed stable, interpretable motion without the previous violent
oscillation, empty grasp loops, or systematic grasp-and-drop behavior.

### 11.2 Formal seed 20260901

- Task success: 111 / 120 = 92.5%
- Clean success: 111 / 120 = 92.5%
- Wrong-object contact: 0%
- Target class accuracy: 100%

Per-task clean success:

- Pick-A: 18 / 20 = 90%
- Pick-B: 20 / 20 = 100%
- Pick-C: 20 / 20 = 100%
- Push-A: 19 / 20 = 95%
- Push-B: 18 / 20 = 90%
- Push-C: 16 / 20 = 80%

Failures were seven insufficient pushes, one object drop, and one missed grasp.

### 11.3 Formal seed 20261017

- Task success: 114 / 120 = 95.0%
- Clean success: 113 / 120 = 94.17%
- Wrong-object contact: 3 / 120 = 2.5%
- Target class accuracy: 100%

Per-task clean success:

- Pick-A: 19 / 20 = 95%
- Pick-B: 20 / 20 = 100%
- Pick-C: 19 / 20 = 95%
- Push-A: 18 / 20 = 90%
- Push-B: 20 / 20 = 100%
- Push-C: 17 / 20 = 85%

Failures were three wrong-object contacts, three insufficient pushes, and one
insufficient lift. The Pick-C insufficient-lift label corresponded to a
toppled, ungrasped cylinder and is better interpreted as a failure-taxonomy
limitation.

### 11.4 Combined 240-episode standard

| Task | Clean successes | Rate |
|---|---:|---:|
| Pick-A | 37 / 40 | 92.5% |
| Pick-B | 40 / 40 | 100% |
| Pick-C | 39 / 40 | 97.5% |
| Push-A | 37 / 40 | 92.5% |
| Push-B | 38 / 40 | 95.0% |
| Push-C | 33 / 40 | 82.5% |
| **Overall** | **224 / 240** | **93.33%** |

Combined clean-failure distribution:

- Insufficient push distance: 10
- Wrong-object contact: 3
- Object dropped: 1
- Missed grasp: 1
- Insufficient lift / toppled before grasp: 1

The combined result exceeds the 80% project gate by 13.33 percentage points.
The lower bound of the approximate 95% Wilson interval is 89.45%, also well
above the target.

## 12. Known Residual Limitations

### 12.1 Push-C remains the weakest task

Push-C achieved 82.5% across 40 episodes. Most failures reached the target and
moved it but stopped before 6.5 cm. Logs show premature transitions from Phase
7 to Phase 8, followed by small hold actions.

The likely architectural cause is limited long-term progress memory. Five
state frames describe current motion but do not explicitly preserve the
object's initial position or remaining displacement to the goal. The model can
therefore infer target identity and contact correctly while estimating task
completion from appearance and duration.

### 12.2 Push actions remain aggressive

Across the combined tests, mean action clipping was highest for Push-B and
workspace/table-floor intervention was highest for Push-C. Successful episodes
also had high clipping rates, so clipping was not the main failure cause. It is
still a useful robustness signal for future work.

### 12.3 Post-failure phase confusion

Rare Pick failures can leave the policy in an out-of-distribution state where
it predicts Push phases 6 or 7 despite a Pick instruction. The final release
does not hide this with a task-conditioned runtime mask because the baseline is
intended to remain raw. Future architectures may condition phase support more
strongly on the language task during training.

### 12.4 Failure taxonomy can be refined

A toppled, ungrasped cylinder can currently be labeled `insufficient_lift` if
its center briefly rises above the height heuristic. Future evaluators should
check `grasped_once` and uprightness before assigning a lift-related category.

## 13. Why V2 Clean Is Frozen as the Final Standard

V2 Clean is accepted as the standard baseline because it satisfies all of the
following simultaneously:

1. Two independent 120-episode clean evaluations exceed 92%.
2. The combined 240-episode clean rate is 93.33%.
3. Every task exceeds or equals 82.5% across 40 episodes.
4. Target classification is 100% in both formal runs.
5. Pick tasks achieve 96.67% combined over 120 Pick episodes.
6. The old systemic grasp-without-lift failure is eliminated.
7. Evaluation uses raw model actions without simulator-state assistance.
8. Dataset, model, optimizer, and checkpoint Preflight passes.
9. Full-trajectory validation and test Postflight passes.
10. Visual and non-rendered evaluations agree closely.

The policy should therefore be preserved as the clean reference before visual,
physics, camera, language, and initial-state robustness perturbations are
introduced. Future improvements should be compared against this frozen
checkpoint rather than silently replacing it.

## 14. Engineering Lessons

1. Clean causal labels matter more than adding recovery complexity.
2. A grasp detector and a lift action must be validated jointly, not as
   independent metrics.
3. Initial-state and transition-specific gates catch failures hidden by mean
   validation loss.
4. Runtime assistance is useful for diagnosis but invalidates a raw policy
   baseline if it uses privileged simulator state.
5. Visual inspection remains essential for physical semantics such as pushing
   upward, toppling, or contacting a corner.
6. Failed expert attempts should be retained only when the successful recovery
   suffix is unambiguous and causally labeled.
7. Schema validators should encode behavioral facts, not only tensor shapes.
8. Multiple evaluation seeds are necessary before declaring a final success
   rate.
9. Large data collection should begin only after a smaller oracle benchmark and
   preflight pass.
10. Once a baseline clears its objective with margin, freeze it before pursuing
    more ambitious robustness features.
