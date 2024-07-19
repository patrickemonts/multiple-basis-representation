.PHONY: clean, default

default:
	echo "There is no default target here. Try clean"

clean:
	${RM} *.log
	${RM} *.csv
