#!/usr/bin/env bash
set -e
#set -x
# $1: model name (in models/)
# $2: input path
# $3: output path

function usage()
{
    echo ""
    echo ""
    echo -e "./predict.sh"
    echo -e "\t-h --help"
    echo -e "\t--input=$INPUTDIR"
    echo -e "\t--output=$OUTPUTDIR"
    echo -e "\t--param:model=$PARAMS"
    echo ""
}

OPTS=`getopt -o i:o:p: --long input:,output:,param:model: -n 'parse-options' -- "$@"`

if [ $? != 0 ] ; then echo "Failed parsing options." >&2 ; exit 1 ; fi

echo "$OPTS"
eval set -- "$OPTS"



while true; do
  case "$1" in
    -i | --input ) INPUTDIR="$2"; shift; shift ;;
    -o | --output ) OUTPUTDIR="$2"; shift; shift ;;
    -p | --param:model ) PARAMS="$2"; shift; shift ;;
    -- ) shift; break ;;
    * ) break ;;
  esac
done

if [ -z "$PARAMS" ]; then
    PARAMS="full_model"
fi
echo "input_dir is $INPUTDIR";
echo "output_dir is $OUTPUTDIR";
echo "param is $PARAMS";

python /src/train_rnn.py preprocessing_predict ddi full_model $INPUTDIR $OUTPUTDIR words wordnet common_ancestors concat_ancestors
