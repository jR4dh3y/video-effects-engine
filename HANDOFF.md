# Handoff: engine and general-purpose editor discussion

Prepared on 2026-10-08 for Radhey and a collaborator. This is a curated handoff of the relevant conversation, not a raw export of private agent history.

## Radhey's objective

Share the existing video effects code so a friend can download it and integrate it into another video engine. Radhey is also considering a general-purpose editor built around these ideas. No full editor implementation or framework decision has been approved.

## Existing work

The original CS2 project built eight montage revisions using `engine.py`, `mix.py`, `assemble.py`, and handwritten Python edit builders that emitted JSON. The renderer performs smooth output-to-source time mapping, RIFE interpolation, optical-flow blur, stepped fast-forward, and impact effects. Music and source audio are mixed separately; assembly adds cross-dissolves and exports the final file.

This repository extracts the three runtime scripts. It replaces machine-specific paths with portable defaults and environment settings, defaults to CPU encoding, makes the RIFE GPU configurable, disables original watermark boxes by default, and checks the video encoder exit status. It does not include original media, analysis results, render caches, or historical edit builders.

## What the Claude discussion said

The workspace conversation compared this engine with HyperFrames, then referred to the original montage conversation and a dedicated generalization review. Claude considered the engine useful as a reusable montage/effects backend, with a declarative timeline, analyze/plan/render CLI, event-to-beat planning, fast preview, configurable media handling, and performance work.

Claude described HyperFrames as better suited to browser-based graphics and overlays. That comparison was based on skill descriptions, not a tested side-by-side evaluation. Do not treat claims about its limitations as verified current product facts.

The generalization review called the conversion mostly packaging. The subsequent discussion challenged that estimate: a general-purpose interactive editor needs considerably more infrastructure than a montage CLI.

## Lessons from the montage revisions

- Vision-model timestamps and kill-feed detection were unreliable enough to require manual frame checks.
- Radhey repeatedly corrected late impacts, incorrect beat choices, and sequences made too fast merely to fit more kills.
- Preserve connected actions and natural pacing; adding more footage must not automatically accelerate everything.
- Audio and visual timing need separate checks. Matching a timestamp is not proof that an edit feels synchronized.
- Several handoffs explicitly lacked real-time viewing and listening. Frame sheets and loudness measurements did not settle subjective pacing issues.
- RIFE concurrency exhausted a 4 GB GPU; raw caches also filled disk. Preview and resource budgeting matter.

## Proposed editor direction

Retain the time-remapping and effects ideas as a render backend. Build a general timeline with arbitrary clips and tracks. Treat montage styling as an optional preset. Rename kill concepts to generic event markers when redesigning the public project format, and make automatic black-and-white, shake and flash choices explicit effects.

A focused first editor release would support importing media, trimming, splitting, arranging and layering clips; playback with sound and precise scrubbing; undo; transform, opacity, volume and effect keyframes; speed ramps; project saving; and reliable export.

AI should propose inspectable timeline edits that the user can change. Detection uncertainty should remain visible rather than silently becoming final timing.

## First proof to build

Two video layers, one music track, one speed ramp, and one dissolve, with preview and export agreeing on timing and appearance. This is a proposed next step, not functionality already present.

The unresolved product choice is a conventional timeline editor with AI assistance versus a prompt-driven editor whose timeline is mainly used for correction. Radhey has not answered that question yet.

## Validation and coverage

Applies: video renderer, external RIFE integration, audio mixer, assembly, example timeline, dependency setup, licensing, GitHub delivery, and this handoff.

Does not apply: browser UI, mobile clients, server deployment, PR workflow, or a full editor implementation. Those are outside this sharing task.

See `VALIDATION.md` for checks performed on the extracted package and remaining untested paths.
