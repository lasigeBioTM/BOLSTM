#!/usr/bin/env bash
set -e
#set -x
# $1: model name (in models/)
# $2: input path
# $3: output path
# based on https://gist.github.com/jehiah/855086


ENVIRONMENT="dev"
DB_PATH="/data/db"

function usage()
{
    echo "if this was a real script you would see something useful here"
    echo ""
    echo -e "./predict.sh"
    echo -e "\t-h --help"
    echo -e "\t--input=$INPUTDIR"
    echo -e "\t--output=$OUTPUTDIR"
    echo -e "\t--param:model=$PARAMS"
    echo ""
}

while [ "$1" != "" ]; do
    PARAM=`echo $1 | awk -F= '{print $1}'`
    VALUE=`echo $1 | awk -F= '{print $2}'`
    case $PARAM in
        -h | --help)
            usage
            exit
            ;;
        --input)
            INPUTDIR=$VALUE
            ;;
        --output)
            OUTPUTDIR=$VALUE
            ;;
        --param:model)
            PARAMS=$VALUE

            ;;
        *)
            echo "ERROR: unknown parameter \"$PARAM\""
            usage
            exit 1
            ;;
    esac
    shift
done
if [ -z "$PARAMS" ]; then
    PARAMS="full_model"
fi
#echo "input_dir is $INPUTDIR";
#echo "output_dir is $OUTPUTDIR";
#echo "params is $PARAMS";

python /src/train_rnn.py preprocessing_predict ddi full_model $INPUTDIR $OUTPUTDIR words wordnet common_ancestors concat_ancestors
