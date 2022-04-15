t0 = time.time()
    # tm = torch.cuda.get_device_properties(0).total_memory
    # print('Total mem: ', tm)
    # m0 = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    # time.sleep(2)
    ort_sess = init_gender_model()
    dataset = Dataset(path = './results/restored_imgs', ort= ort_sess)
    train_ldr = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
    # m1 = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    # print('memory usage by dataloader :', tm - m0 - m1)
    output_dict = {}
    age_model = init_age_model()
    ort_sess = init_gender_model()
    m2 = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
    # print('Mem on loading model: ',m2-m1, 'Mem from inital:', tm-m0-m1-m2)

    for batch, data in enumerate(tqdm(train_ldr)):
        #'efficientnetv2_s_input:0'
        # print(batch, data[1]['efficientnetv2_s_input:0'].shape, data[0].shape)
        torch.cuda.empty_cache()
        age_input = data[0]
        gender_input = {ort_sess.get_inputs()[0].name : data[1]['efficientnetv2_s_input:0'].squeeze().detach().numpy()}     
        # age_output = age_regression(age_input, age_model)
        gender_output = gender_detect(gender_input, ort_sess)
        # del age_output
        del gender_output
        # m3 = torch.cuda.memory_reserved(0) - torch.cuda.memory_allocated(0)
        # print('Mem on processing: ',m3-m2, 'Mem from inital:', tm-m3-m0-m1-m2)
        # print(age_output.shape)
        # print(gender_output[0].shape)
        # break
        # output_dict[batch]
    print('Dataset creation time:' , time.time() - t0)



