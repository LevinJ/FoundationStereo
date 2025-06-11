"""
<!-- ******************************************
*  Author : Levin Jian  
*  Created On : Wed Jun 11 2025
*  File : combine_annotation.py
******************************************* -->

"""

import os
import json

class CombineAnnotation(object):
    def __init__(self):
        return
    def run(self):
        annotation_files = [
            '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/jiuting_campus/annotation/zed_annotation.json',
            '/media/levin/DATA/nerf/new_es8/stereo/250610/annotation/zed_annotation.json'
        ]
        merged_items = []
        for ann_path in annotation_files:
            with open(ann_path, 'r') as f:
                ann_dict = json.load(f)
            ann_dir = os.path.dirname(ann_path)
            ann_list = ann_dict.get('files', [])
            for item in ann_list:
                # Fix rgb path
                if not os.path.isabs(item.get('rgb', '')):
                    item['rgb'] = os.path.normpath(os.path.join(ann_dir, item['rgb']))
                # Fix depth path
                if not os.path.isabs(item.get('depth', '')):
                    item['depth'] = os.path.normpath(os.path.join(ann_dir, item['depth']))
                merged_items.append(item)
        merged_annotations = {'files': merged_items}
        # Ensure temp directory exists under the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        out_path = os.path.join(temp_dir, 'merged_zed_annotation.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_annotations, f, indent=2)
        print(f"Merged annotation saved to {out_path}")

if __name__ == "__main__":   
    obj= CombineAnnotation()
    obj.run()
