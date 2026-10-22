MAIN := main
TEX := latexmk -pdf
MON_OPT := -pvc
CLN_OPT := -CA

.PHONY: all monitor main-monitor wordcount clean

all: ${MAIN}.pdf

${MAIN}.pdf:
	${TEX} ${MAIN}.tex

monitor: main-monitor

main-monitor:
	${TEX} ${MON_OPT} ${MAIN}.tex

clean:
	${TEX} ${CLN_OPT} ${MAIN}.tex

wordcount:
	detex -l -n -w ${MAIN}.tex | wc -w
