CREATE_IMAGE_QUERY = """
                        INSERT INTO images (filename, file, predictions, ground_truth)
                        VALUES (:filename, :file, :predictions, :ground_truth)
                        RETURNING id, filename, file, predictions, ground_truth;
                     """

GET_IMAGE_BY_ID_QUERY = """
                           SELECT *
                           FROM images
                           WHERE id = :id;
                        """

LIST_ALL_IMAGES_QUERY = """
                           SELECT *
                           FROM images
                        """

UPDATE_IMAGE_PREDICTIONS_QUERY = """
                                    UPDATE images
                                    SET predictions = :predictions
                                    WHERE id = :id
                                    RETURNING id, filename, file, predictions, ground_truth;
                                 """

DELETE_IMAGE_BY_ID_QUERY = """
                              DELETE FROM images
                              WHERE id = :id
                              RETURNING id;
                           """
