"""生成 FROSS class mapping JSON。"""
import argparse, json, os

VG_CLASSES = [
    'airplane','animal','arm','bag','banana','basket','bathtub','bear','bed',
    'bench','bike','bird','board','boat','book','boot','bottle','bowl','box',
    'boy','branch','building','bus','cabinet','cap','car','cat','chair',
    'child','clock','coat','counter','cow','cup','curtain','desk','dog','door',
    'drawer','ear','elephant','engine','eye','face','fence','finger','flag',
    'flower','food','fork','fruit','giraffe','girl','glass','glove','guy',
    'hair','hand','handle','hat','head','helmet','hill','horse','house',
    'jacket','jean','kid','kite','lady','lamp','laptop','leaf','leg','letter',
    'light','logo','man','men','mirror','motorcycle','mountain','mouth','neck',
    'nose','number','orange','pant','paper','paw','people','person','phone',
    'pillow','pizza','plane','plant','plate','player','pole','post','pot',
    'racket','railing','ring','rock','roof','room','screen','seat','shelf',
    'shirt','shoe','short','sidewalk','sign','sink','skateboard','ski','skirt',
    'sky','snow','sock','sports','street','student','surfboard','table','tail',
    'tie','tile','tire','toilet','towel','tower','track','train','tree',
    'truck','trunk','umbrella','van','vase','vegetable','vehicle','wall',
    'wheel','window','windshield','wing','wire','woman','zebra'
]
VG_RELS = [
    'above','across','against','along','and','at','attached to','behind',
    'belonging to','between','carrying','covered in','covering','eating',
    'flying in','for','from','growing on','hanging from','has','holding',
    'in','in front of','laying on','looking at','lying on','made of',
    'mounted on','near','of','on','on back of','over','painted on',
    'parked on','part of','playing','riding','says','sitting on',
    'standing on','to','under','using','walking in','walking on',
    'watching','wearing','wears','with'
]

SR_CLASSES = ['bathtub','bed','bookshelf','cabinet','chair','counter','curtain',
              'desk','door','floor','otherfurniture','picture','refridgerator',
              'shower curtain','sink','sofa','table','toilet','wall','window']
SR_RELS = ['attached to','build in','connected to','hanging on',
           'part of','standing on','supported by']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["vg", "3rscan"], default="vg")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.model == "3rscan":
        classes = SR_CLASSES + [f'object_{i}' for i in range(20, 150)]
        rels = SR_RELS + [f'rel_{i}' for i in range(7, 50)]
    else:
        classes = VG_CLASSES
        while len(classes) < 150: classes.append(f'object_{len(classes)}')
        classes = classes[:150]
        rels = VG_RELS
        while len(rels) < 50: rels.append(f'rel_{len(rels)}')
        rels = rels[:50]

    mapping = {"VisualGenome_list": classes, "VisualGenome_rel": rels}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(mapping, f)
    print(f"{len(classes)} classes, {len(rels)} relations → {args.output}")


if __name__ == "__main__":
    main()
