# Security Policy

## Supported Versions

We actively support the following versions of HeavyTails with security updates:

Version | Supported
------- | ------------------
0.1.x   | :white_check_mark:
< 0.1   | :x:

## Reporting a Vulnerability

The HeavyTails team takes security vulnerabilities seriously. We appreciate your efforts to responsibly disclose any security issues you may find.

### How to Report

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report security vulnerabilities to us via email:

- **Email**: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- **Subject**: [SECURITY] HeavyTails Vulnerability Report
- **PGP Key**: Available upon request

### What to Include

When reporting a vulnerability, please include as much of the following information as possible:

1. **Type of issue** (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
2. **Full paths of source file(s)** related to the manifestation of the issue
3. **Location of the affected source code** (tag/branch/commit or direct URL)
4. **Special configuration required** to reproduce the issue
5. **Step-by-step instructions** to reproduce the issue
6. **Proof-of-concept or exploit code** (if possible)
7. **Impact of the issue**, including how an attacker might exploit it

### Response Timeline

We aim to respond to security vulnerability reports within:

- **Initial Response**: 48 hours
- **Assessment**: 1 week
- **Fix Development**: 2-4 weeks (depending on complexity)
- **Release**: As soon as fix is ready and tested

### Security Update Process

1. **Vulnerability Assessment**: We will assess the reported vulnerability and its impact
2. **Fix Development**: We will develop a fix for the vulnerability
3. **Testing**: The fix will be thoroughly tested
4. **Release**: A security release will be published
5. **Disclosure**: We will publicly disclose the vulnerability after the fix is released

## Security Considerations

### Code Execution Risks

HeavyTails is a mathematical library that processes numerical data. However, users should be aware of potential security considerations:

1. **Input Validation**: Always validate input data from untrusted sources
2. **Memory Usage**: Large datasets or extreme parameters may cause excessive memory usage
3. **Numerical Stability**: Invalid parameters may cause infinite loops or crashes
4. **File I/O**: Be cautious when reading data files from untrusted sources

### Safe Usage Guidelines

#### Input Validation

```python
# Good: Validate parameters
def safe_pareto_analysis(data, alpha, xm):
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError("Alpha must be positive")
    if not isinstance(xm, (int, float)) or xm <= 0:
        raise ValueError("xm must be positive")
    # ... rest of analysis
```

#### Resource Limits

```python
# Good: Set reasonable limits
MAX_SAMPLE_SIZE = 10_000_000
MAX_ITERATIONS = 100_000

def safe_sampling(distribution, n):
    if n > MAX_SAMPLE_SIZE:
        raise ValueError(f"Sample size too large: {n} > {MAX_SAMPLE_SIZE}")
    return distribution.rvs(n)
```

#### File Handling

```python
# Good: Validate file inputs
def safe_read_data(filepath):
    # Check file size
    if filepath.stat().st_size > 100_000_000:  # 100MB limit
        raise ValueError("File too large")

    # Validate file extension
    if filepath.suffix not in {'.csv', '.txt', '.json'}:
        raise ValueError("Unsupported file type")

    # Read with limits
    with open(filepath) as f:
        return [float(line.strip()) for line in f if line.strip()]
```

## Known Security Considerations

### Mathematical Operations

1. **Division by Zero**: Some distributions may have singularities
2. **Overflow/Underflow**: Extreme parameters can cause numerical overflow
3. **Infinite Loops**: Poorly conditioned parameters in iterative algorithms
4. **Memory Exhaustion**: Large sample generation or parameter fitting

### Mitigation Strategies

- **Parameter Validation**: All distributions validate input parameters
- **Numerical Limits**: Reasonable bounds on iterations and computations
- **Error Handling**: Graceful handling of edge cases
- **Resource Management**: Memory-efficient algorithms where possible

## Dependencies Security

HeavyTails is designed to be dependency-free (pure Python) to minimize security attack surface:

- **No external dependencies** in the core library
- **Optional dependencies** only for development, testing, and examples
- **Regular updates** of development dependencies through automated tools

## Vulnerability Disclosure Timeline

We follow responsible disclosure practices:

1. **Private notification** to the development team
2. **Fix development and testing** (typically 2-4 weeks)
3. **Security release** with fix
4. **Public disclosure** 7 days after release (or by mutual agreement)

## Security Hall of Fame

We will recognize security researchers who responsibly report vulnerabilities:

_No security vulnerabilities have been reported yet._ 

<!-- Template for future entries: - **[Researcher Name]** - [Date] - [Brief description of vulnerability type] -->

## Contact Information

For security-related inquiries:

- **Security Email**: [dfr@esmad.ipp.pt](mailto:dfr@esmad.ipp.pt)
- **Maintainer**: Diogo Ribeiro
- **Response Time**: Typically within 48 hours

## Security Tools

We use automated security tools in our CI/CD pipeline:

- **Bandit**: Python security linter
- **Safety**: Dependency vulnerability scanner
- **CodeQL**: Semantic code analysis
- **pip-audit**: Python package vulnerability scanner

## Security Best Practices

When using HeavyTails in production environments:

1. **Input Sanitization**: Validate all external inputs
2. **Resource Limits**: Set appropriate limits for computational resources
3. **Error Handling**: Implement proper error handling for edge cases
4. **Logging**: Log security-relevant events appropriately
5. **Updates**: Keep the library updated to the latest version

--------------------------------------------------------------------------------

**Note**: This security policy applies to the HeavyTails library itself. Users are responsible for the security of their own applications that use this library.
