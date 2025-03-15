whitespace:
	find . -name "*.py" -exec sed -i '' 's/^[[:space:]]*$//g' {} \;