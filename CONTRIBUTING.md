# Contributing Guidelines

Thank you for your interest in contributing to the Biceps Brachii Ultrasound Segmentation project. We welcome academic, scientific, and non-commercial open-source contributions.

## Code of Conduct

All contributors, researchers, and users are expected to maintain professional, constructive, and respectful communication in all interactions across issues, pull requests, and project discussions.

## Non-Commercial License Compliance

This project is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license. 
By contributing to this repository:
1. You agree that your contributions will be licensed under the CC BY-NC-SA 4.0 license.
2. You confirm that any code, data, annotations, or weights submitted are your original work or properly attributed under compatible non-commercial terms.
3. You acknowledge that no commercial deployment or proprietary redistribution is permitted without explicit written agreement from the primary author (Cornelio, cornelioabdimash@gmail.com).

## How to Contribute

### Reporting Bugs and Suggesting Features
- Search existing issues before creating a new report to avoid duplicates.
- Provide a clear, descriptive issue title and detailed reproduction steps.
- Include Python, PyTorch, CUDA, and operating system version details.
- Attach minimal logs or error stack traces without enclosing personal patient health information (PHI).

### Submitting Pull Requests
1. Fork the repository and create a descriptive feature branch from `master` (`git checkout -b feature/improved-loss-metric`).
2. Ensure all code conforms to PEP 8 standards and is properly documented with type hints.
3. If introducing changes to model architectures or datasets, verify that GPU memory footprint remains compliant on mobile/entry-level GPUs (RTX 2050 / 4GB VRAM).
4. Commit your changes with clear, imperative commit messages (e.g., `feat(model): Add Dice-Focal hybrid loss`).
5. Open a Pull Request referencing any related issues and provide a concise summary of the changes and validation results.

## Contact

For questions regarding research collaboration, data sharing, or licensing, please contact:
Cornelio (cornelioabdimash@gmail.com)
