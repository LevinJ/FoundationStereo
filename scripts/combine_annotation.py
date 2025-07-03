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
        self.use_rel_path = True  # Use relative paths for RGB and depth files
        self.base_dir = '/media/levin/DATA/nerf/new_es8/stereo/annotations'  # Base directory for annotations
        return
    def run(self):
        # annotation_files = [
        #     '/media/levin/DATA/nerf/new_es8/stereo_20250331/20250331/jiuting_campus/annotation/zed_annotation.json',
        #     '/media/levin/DATA/nerf/new_es8/stereo/250610/annotation/zed_annotation.json'
        # ]

        annotation_files = [
            '/media/levin/DATA/nerf/new_es8/stereo/annotations/20250702/zed_annotation_eval.json',
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
                if self.use_rel_path:
                    item['rgb'] = os.path.relpath(item['rgb'], self.base_dir)
                    
                # Fix depth path
                if not os.path.isabs(item.get('depth', '')):
                    item['depth'] = os.path.normpath(os.path.join(ann_dir, item['depth']))
                if self.use_rel_path:
                    item['depth'] = os.path.relpath(item['depth'], self.base_dir)
                merged_items.append(item)
        merged_annotations = {'files': merged_items}
        # Ensure temp directory exists under the script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        temp_dir = os.path.join(script_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        anno_file_sufix = ''
        if '_eval' in annotation_files[0]:
            anno_file_sufix = '_eval'
        out_path = os.path.join(temp_dir, f'merged_zed_annotation{anno_file_sufix}.json')
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(merged_annotations, f, indent=2)
        print(f"Merged annotation saved to {out_path}")

if __name__ == "__main__":   
    obj= CombineAnnotation()
    obj.run()
