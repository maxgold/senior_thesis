#!/bin/bash
for filename in *.obj; do
	cp template.urdf "${filename%.obj}.urdf"
	sed -i -e "s/duck.obj/$filename/g" "${filename%.obj}.urdf"
done

for filename in *.urdf; do
	sed -i -e 's/scale="1[[:space:]]1[[:space:]]1"/scale="2 2 2"/g' "$filename"
done

filename=vase.obj
