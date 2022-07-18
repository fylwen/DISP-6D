import os
import pickle

import matplotlib.pyplot as plt

#Code for plotting the pose evaluation curves on real275 for Ours-per (setting I) and Ours-all (setting III)
#w.r.t. AP at different rotation error, translation error and 3D IoU thresholds
#The plotted data are from the pkl files under the ./ws/real275_curve/ folder
#while the results are saved to the ./ws/real275_curve/curves/ folder.



def draw_pose_ap(synset_names, log_dir,  tag_methods, legend_methods, colors, styles):
    # draw pose AP vs. thresholds
    num_classes = len(synset_names)
    degree_thres_list = list(range(0, 61, 1))+ [360]
    shift_thres_list = list(range(0,16,1)) + [100]


    pose_dicts = []
    for ctm in tag_methods:
        pose_dict_pkl_path =  os.path.join(log_dir, ctm+'_rt_err.pkl')
        with open(pose_dict_pkl_path, 'rb') as f:
            cpd = pickle.load(f)
            pose_dicts.append(cpd)


    legend_font = {'family': 'serif', 'weight': 'bold', 'size': 24 }
    line_width=[6,5,5,5,5,5,5]

    for cls_id in range(0, num_classes):
        class_name = synset_names[cls_id] if cls_id>0 else 'Mean'
        #for rotation error
        if cls_id==0:
            fig_rot = plt.figure(figsize=(6,6),facecolor='white')
            plt.ylabel('Average Precision',fontsize=28, family='serif', weight='bold')
            plt.xlabel('Rotation/degree',fontsize=28, family='serif',  weight='bold')
            plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=22,family='serif', weight='bold')
            plt.xticks([0,10,20,30,40,50,60],fontsize=22,family='serif',  weight='bold')
            plt.title(class_name, loc='center',fontsize=28, family='serif',  weight='bold')
            plt.grid(linestyle='-.')
        else:
            fig_rot = plt.figure(figsize=(3,3),facecolor='white')
            plt.yticks([0,0.5,1.0],fontsize=18,family='serif', weight='bold')
            plt.xticks([0,20,40,60],fontsize=18,family='serif', weight='bold')
            plt.title(class_name.lower(), loc='center', fontsize=24,family='serif', weight='bold')


        plt.ylim((0, 1.01))
        plt.xlim((0, degree_thres_list[-2]))

        plt.tight_layout()

        for mid in range(0, len(tag_methods)):
            pose_aps = pose_dicts[mid]['aps']
            method_name=legend_methods[mid]
            if cls_id==0: #Mean
                plt.plot(degree_thres_list[:-1], pose_aps[-1, :-1, -1],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)
            else:
                plt.plot(degree_thres_list[:-1], pose_aps[cls_id, :-1, -1],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)


        if cls_id==0:
            plt.legend(loc='lower right',bbox_to_anchor=(1.05,-0.05),frameon=False, prop=legend_font,markerscale=0.5, handletextpad=0.3)
        
        if not os.path.exists(os.path.join(log_dir,'curves')):
            os.makedirs(os.path.join(log_dir,'curves'))
        output_path = os.path.join(log_dir, 'curves', 'err_rot_{:s}.png'.format(class_name))
        fig_rot.savefig(output_path)
        plt.close(fig_rot)

        #for translation error
        if cls_id==0:
            fig_tra = plt.figure(figsize=(6,6),facecolor='white')
            plt.ylabel('Average Precision',fontsize=28, family='serif', weight='bold')
            plt.xlabel('Translation/cm',fontsize=28, family='serif',  weight='bold')
            plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=22,family='serif', weight='bold')
            plt.xticks([0,3,6,9,12,15],fontsize=22,family='serif',  weight='bold')
            plt.title(class_name, loc='center',fontsize=28, family='serif',  weight='bold')
            plt.grid(linestyle='-.')
        
        else:
            fig_tra = plt.figure(figsize=(3,3),facecolor='white')
            plt.yticks([0,0.5,1.0],fontsize=18,family='serif', weight='bold')
            plt.xticks([0,5,10,15],fontsize=18,family='serif', weight='bold')
            plt.title(class_name.lower(), loc='center', fontsize=24,family='serif', weight='bold')


        plt.ylim((0, 1.01))
        plt.xlim((0,shift_thres_list[-2]))
        plt.tight_layout()
        
        for mid in range(0, len(tag_methods)):
            pose_aps = pose_dicts[mid]['aps']
            method_name=legend_methods[mid]
           
            if cls_id==0:
                plt.plot(shift_thres_list[:-1], pose_aps[-1, -1, :-1],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)
            else:
                plt.plot(shift_thres_list[:-1], pose_aps[cls_id, -1, :-1],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)


        if cls_id==0:
            plt.legend(loc='lower right', bbox_to_anchor=(1.05, -0.05), frameon=False, prop=legend_font, markerscale=0.5, handletextpad=0.3)

        output_path = os.path.join(log_dir, 'curves', 'err_tra_{:s}.png'.format(class_name))
        fig_tra.savefig(output_path)
        plt.close(fig_tra)



def draw_iou_ap(synset_names, log_dir, tag_methods,legend_methods, colors,styles):
    # draw pose AP vs. thresholds

    num_classes = len(synset_names)
    iou_dicts = []
    for ctm in tag_methods:
        pose_dict_pkl_path = os.path.join(log_dir, ctm + '_iou.pkl')
        with open(pose_dict_pkl_path, 'rb') as f:
            cpd = pickle.load(f)
            iou_dicts.append(cpd)

    legend_font = {'family': 'serif', 'weight': 'bold', 'size': 24 }
    line_width=[6,5,5,5,5,5,5]

    for cls_id in range(0, num_classes):
        class_name = synset_names[cls_id] if cls_id>0 else 'Mean'

        if cls_id==0:
            fig_iou = plt.figure(figsize=(6,6),facecolor='white')
            plt.ylabel('Average Precision',fontsize=28, family='serif', weight='bold')
            plt.xlabel('3D IoU',fontsize=28, family='serif',  weight='bold')
            plt.yticks([0,0.2,0.4,0.6,0.8,1.0],fontsize=22,family='serif', weight='bold')
            plt.xticks([0,0.25,0.5,0.75,1.0],fontsize=22,family='serif',  weight='bold')
            plt.title(class_name, loc='center',fontsize=28, family='serif',  weight='bold')
            plt.grid(linestyle='-.')
        else:
            fig_iou = plt.figure(figsize=(3,3),facecolor='white')
            plt.yticks([0,0.5,1.0],fontsize=18,family='serif', weight='bold')
            plt.xticks([0,0.5,1.0],fontsize=18,family='serif',  weight='bold')
            plt.title(class_name.lower(), loc='center',fontsize=24, family='serif',  weight='bold')
            
            
        plt.ylim((0, 1.01))
        plt.xlim((0, 1.01))
        plt.tight_layout()

        for mid in range(0, len(tag_methods)):
            iou_thres_list=iou_dicts[mid]['thres_list']
            iou_aps = iou_dicts[mid]['aps']

            method_name=legend_methods[mid]

            if cls_id==0:
                plt.plot(iou_thres_list, iou_aps[-1, :],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)

            else:
                plt.plot(iou_thres_list, iou_aps[cls_id, :],color=colors[mid],ls=styles[mid],linewidth=line_width[cls_id], label=method_name)

        if cls_id==0:
            plt.legend(loc='lower left',prop=legend_font)
        if not os.path.exists(os.path.join(log_dir,'curves')):
            os.makedirs(os.path.join(log_dir,'curves'))
        output_path = os.path.join(log_dir, 'curves','iou3d_{:s}.png'.format(class_name))
        fig_iou.savefig(output_path)
        plt.close(fig_iou)






synset_names = ['BG',  # 0
                'Bottle',  # 1
                'Bowl',  # 2
                'Camera',  # 3
                'Can',  # 4
                'Laptop',  # 5
                'Mug'  # 6
                ]
log_dir = './ws/real275_curve'

draw_pose_ap(synset_names,log_dir,
    tag_methods=['ours_per','ours_all'],
    legend_methods=['Ours-per','Ours-all'],
    colors=['#4169E1','#4169E1'],
    styles=['-','--'])

draw_iou_ap(synset_names,log_dir,
    tag_methods=['ours_per','ours_all'],
    legend_methods=['Ours-per','Ours-all'],
    colors=['#4169E1','#4169E1'],
    styles=['-','--'])
