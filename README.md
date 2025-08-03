# ZaoAI – Z Anime Opening AI

![alt text](zaoai/img/showcase2.png)

> ⚠️ This project is in early development.

## Project Goal
Input an anime video file and automatically mark:
- **OP (Opening)** start/end
- **ED (Ending)** start/end  
Based primarily on **audio analysis**.

## Usage
> ⚠️ This project is in early development.
### Zaoai
Working directory: zaoai/zaoai 

### Zaoai-Helper
Working directory: zaoai/zaoai-helper

## Roadmap

### Phase 1 – Dataset Preparation
- [ ] Zaoaihelper to extract training data
- [ ] Chapter annotation tools for OP/ED regions

### Phase 2 – Training Pipeline
- Use sliced FFTs of audio as input
- Train using labels of frame numbers to identify opening start/end segments

## Crates
> ⚠️ This project is in early development.
### zaoai
Main crate, contains neural network code and gui for interaction

### zaoai-types
Crate with common datatypes & functionality, currently stuff is just crammed in here.

### zaoai-helper
Contains script-like executables for preparing & maintaining files needed for zaoai. Such as converting .mkv files to .zlbl/.spectrogram files
