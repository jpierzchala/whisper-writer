# AI Agent Instructions

This document contains important instructions for AI agents (GitHub Copilot, OpenAI Codex, etc.) working on this codebase.

## 🌍 Language Requirement

**IMPORTANT**: All project communications must be in English. This includes:
- Pull request descriptions and titles
- Issue reports and comments
- Code comments and documentation
- Commit messages
- Review comments

This ensures consistency and accessibility for all contributors worldwide.

## 🚨 MANDATORY: Always Run Tests Before Completion

**CRITICAL REQUIREMENT**: Before completing any work or submitting changes, AI agents MUST run the test suite to ensure all tests pass.

### Test Command
```bash
pytest tests/ -v
```

### Test Requirements
- ✅ All 7 tests must pass
- ✅ No test failures or errors allowed
- ✅ If any test fails, fix the issue before completing work

### Current Test Suite
The project contains the following test files:
- `tests/test_failed_audio_saving.py` - Tests for audio data validation and file saving
- `tests/test_failed_audio_simple.py` - Simplified audio validation tests  
- `tests/test_result_thread.py` - Tests for the main ResultThread class

### Test Categories Covered
1. **Audio Data Validation**: Testing `_save_failed_audio()` method with various input scenarios
2. **Transcription Retry Logic**: Testing retry mechanisms when transcription fails
3. **Error Handling**: Testing exception handling in the ResultThread class
4. **File Operations**: Testing failed audio file saving functionality

## Code Quality Standards

### When Making Changes
1. **Always run tests first** to establish baseline
2. Make your changes
3. **Run tests again** to verify nothing is broken
4. Fix any test failures before considering work complete

### Common Issues to Watch For
- Module import conflicts in tests (ensure proper mocking)
- ConfigManager initialization issues
- Audio data validation edge cases
- Thread safety in ResultThread operations

## Test Execution Examples

### Run all tests
```bash
pytest tests/ -v
```

### Run specific test file
```bash
pytest tests/test_result_thread.py -v
```

### Run with coverage (optional)
```bash
pytest tests/ --cov=src -v
```

## Expected Output
When all tests pass, you should see:
```
===== 7 passed in X.XXs =====
```

## ⚠️ NEVER Complete Work With Failing Tests

If tests are failing:
1. Investigate the failure
2. Fix the underlying issue
3. Re-run tests to confirm fix
4. Only then consider the work complete

## 🔧 Environment Setup

**CRITICAL**: Before starting any work, install project dependencies to avoid import errors during testing.

### Recommended Installation Method
```bash
# Try to install the full project (may fail on some platforms due to OS-specific deps)
pip install .
```

### If Full Installation Fails
Install core dependencies needed for development and testing:
```bash
pip install pytest openai requests PyYAML numpy black flake8
```

### Why This Is Essential
- **Missing modules cause test failures**: AI agents often encounter `ModuleNotFoundError` when trying to run tests
- **pyproject.toml includes all dependencies**: The project correctly defines all necessary modules
- **Platform-specific dependencies may fail**: Some dependencies (like `pywin32`) are OS-specific and may cause installation errors on non-Windows systems
- **Core dependencies are sufficient for most work**: For testing and basic development, the core dependencies listed above are adequate

### Verification
After installation, verify core modules are available:
```bash
python -c "import pytest, openai, requests, yaml, numpy; print('Core dependencies available')"
```

---

**Remember**: Code without passing tests is incomplete code. Always verify your changes don't break existing functionality.
