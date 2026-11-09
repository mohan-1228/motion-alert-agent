# Motion Alert Agent

A local AI agent that detects motion with a Raspberry Pi, filters for people, and sends smart alerts to Discord.

Status: in development (Sprint 0, environment setup).

## Architecture
Raspberry Pi (PIR sensor + camera) -> Mac Mini (person detection + local LLM agent) -> Discord

## Layout
- `pi/`   code that runs on the Raspberry Pi
- `mac/`  code that runs on the Mac Mini
- `docs/` diagrams and notes
