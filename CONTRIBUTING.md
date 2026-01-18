# Contributing Guidelines

## 🎯 Project Vision

Network Intrusion Detection System (NIDS) is a production-quality machine learning system for detecting and classifying network intrusions. We welcome contributions from ML enthusiasts, security researchers, and software engineers.

## 🤝 How to Contribute

### Reporting Issues
1. Check existing issues first
2. Provide detailed description
3. Include error messages and logs
4. Specify Python version and OS

### Suggesting Improvements
1. Open an issue with "Enhancement" label
2. Describe the improvement
3. Explain the benefit
4. Provide example code if possible

### Pull Requests
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m "Add feature description"`
4. Push to branch: `git push origin feature/your-feature`
5. Open a Pull Request

## 📝 Code Style Guidelines

### Python
- Follow PEP 8 conventions
- Use meaningful variable names
- Add docstrings to functions
- Keep functions focused and small
- Use type hints where appropriate

Example:
```python
def train_model(X_train: np.ndarray, 
                y_train: np.ndarray,
                model_type: str = 'xgboost') -> Any:
    """
    Train a machine learning model.
    
    Args:
        X_train: Training features (n_samples, n_features)
        y_train: Training labels (n_samples,)
        model_type: Type of model to train
        
    Returns:
        Trained model object
    """
    # Implementation
    pass
```

### Documentation
- Docstrings for all functions
- Comments for complex logic
- Update README for major changes
- Include examples in docstrings

### Testing
- Test new features before PR
- Verify on different Python versions
- Test with sample datasets
- Check memory usage on large datasets

## 🚀 Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/Network-Intrusion-Detection-System.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install dev dependencies
pip install black flake8 pytest

# Format code
black src/

# Check style
flake8 src/

# Run tests
pytest tests/
```

## 📦 Project Structure

Keep the structure organized:
```
src/
├── preprocessing.py     # Data handling
├── train.py            # Model training
├── explain.py          # Evaluation
└── predict.py          # Inference

tests/
├── test_preprocessing.py
├── test_train.py
└── test_predict.py

docs/
├── API.md
└── ARCHITECTURE.md
```

## 🎓 Contribution Ideas

### Easy (Good for beginners)
- [ ] Add docstring examples
- [ ] Improve comments
- [ ] Update README sections
- [ ] Add usage examples
- [ ] Fix typos

### Medium
- [ ] Add new visualization
- [ ] Implement new metric
- [ ] Add validation checks
- [ ] Improve error messages
- [ ] Add logging

### Hard
- [ ] New ML algorithm
- [ ] Performance optimization
- [ ] Multi-class classification
- [ ] Advanced feature selection
- [ ] Model interpretability features

## 🔄 Review Process

1. **Code Review**: Maintainers review code quality
2. **Testing**: Verify functionality with test data
3. **Documentation**: Check documentation updates
4. **Performance**: Verify no performance degradation
5. **Merge**: Approved PRs are merged

## 📋 Checklist for PRs

- [ ] Code follows PEP 8
- [ ] Added docstrings
- [ ] Updated README if needed
- [ ] Tested on sample data
- [ ] No performance regression
- [ ] Added comments for complex logic
- [ ] All imports are used

## 🐛 Reporting Security Issues

For security vulnerabilities, please email privately instead of using GitHub issues.

## 📄 License

By contributing, you agree to license your contributions under the same license as the project.

## 🙏 Thank You!

Thanks for contributing to Network Intrusion Detection System! 🎉

---

Questions? Open an issue or start a discussion!
